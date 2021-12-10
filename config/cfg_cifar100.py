""" a basic demo of performing classification on the cifar-100 dataset
    SINGLE RUN:
        CUDA_VISIBLE_DEVICES=7 python -m recon.runner.train --cfg config/cfg_cifar100.py
    TWO RUN:
        CUDA_VISIBLE_DEVICES=6,7 python -m recon.runner.train --cfg config/cfg_cifar100.py
"""

BATCH_SIZE_PER_GPU = 64
NUM_EPOCH = 1000
EPOCH_DROP = [0.5, 0.8]
LR_START = 1e-3
LR_DROP_RATE = 0.1
OPTIM_TYPE = "Adam"

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

import torch
from recon.utils import dist_info

rank, num_rank, device = dist_info()

training = dict(epoch_end=NUM_EPOCH)

routine=dict(
    print_every=10,               # per iter
    checkpoint_latest_every=1000, # per iter
    checkpoint_every=1000,        # per iter
    validate_every=1000,          # per iter 
    visualize_every=-1,           # per iter
)

save = dict(
    model_selection_metric="accuracy",
    model_selection_mode="maximize",
)

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

import torchvision.transforms as transforms
from torchvision.datasets import CIFAR100
from recon.utils import dist_samplers, kwargs_shuffle_sampler

kwargs_loader = dict(batch_size=BATCH_SIZE_PER_GPU, num_workers=2)
image_transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
)

dataset_train = CIFAR100(root="./data", train=True,  download=True,  transform=image_transform)
dataset_test  = CIFAR100(root="./data", train=False, download=False, transform=image_transform)
samplers = dist_samplers(dataset_train, dataset_test)
dataloaders = dict(train = torch.utils.data.DataLoader(dataset_train, **kwargs_shuffle_sampler(samplers, "train"), **kwargs_loader),
                   val   = torch.utils.data.DataLoader(dataset_test,  **kwargs_shuffle_sampler(samplers, "val"),   **kwargs_loader))

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

from torchvision.models import resnet18

model = resnet18(num_classes=100)
if num_rank > 1:
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

criterion = torch.nn.CrossEntropyLoss()

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

optimizer = getattr(torch.optim, OPTIM_TYPE)(model.parameters(), lr=LR_START)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, 
                                                 milestones=[int(NUM_EPOCH * rate) for rate in EPOCH_DROP],
                                                 gamma=LR_DROP_RATE)

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

from recon.utils import dict_save_lr

def train_step_fn(batch_data):
    model.train()

    inputs, labels = batch_data
    inputs = inputs.to(device)
    labels = labels.to(device)

    outputs = model(inputs)
    loss = criterion(outputs, labels)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    loss_dict = dict(loss=loss.item())
    dict_save_lr(loss_dict, optimizer=optimizer)
    
    return loss_dict

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

@torch.no_grad()
def evaluate_fn(val_loader):
    model.eval()

    num_correct, num_total = 0, 0
    for batch_data in val_loader:
        inputs, labels = batch_data
        inputs = inputs.to(device)
        labels = labels.to(device)

        preds = model(inputs).argmax(dim=1)

        num_total += len(labels)
        num_correct += (preds == labels).sum().item()
    
    eval_dict = dict(accuracy=num_correct / num_total)
    return eval_dict

