""" a basic demo of classifying some primitive shapes
    SINGLE RUN:
        CUDA_VISIBLE_DEVICES=2 python -m recon.runner.train_ddp --cfg config/cfg_classification.py
    TWO RUN:
        CUDA_VISIBLE_DEVICES=2,3 python -m recon.runner.train_ddp --cfg config/cfg_classification.py
"""

import os
import torch
import trimesh
import numpy as np
from recon.utils import logger_info # print info via logger (saved to file)
from recon.utils import get_tensor_hash
from scipy.spatial.transform import Rotation

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

rank = int(os.environ.get("RANK", "0"))
num_rank = int(os.environ.get("WORLD_SIZE", "1"))
device = torch.device(f"cuda:{rank}")

BATCH_SIZE_PER_GPU = 32
BATCH_SIZE = BATCH_SIZE_PER_GPU * num_rank # total batch size (sum of all processes)
NUM_OBJECTS = 512 # total object num (sum of all processes)
NUM_INPUT_POINTS = 2048

training = dict(
    epoch_end=10_0000,
)

routine=dict(
    print_every=10,               # per iter
    checkpoint_latest_every=100,  # per iter
    checkpoint_every=100,         # per iter
    validate_every=-1,            # per iter 
    visualize_every=-1,           # per iter
)

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

class PrimitiveDataset(torch.utils.data.Dataset):
    def __init__(self, batch_size=16, num_objects=1000, num_points=2048, num_all_points=65536):
        super().__init__()
        assert num_objects % num_rank == 0, "num of objects must be divisible by the process num"
        np.random.seed(0)

        # randomly generate some boxes
        mesh_list = list()
        for i in range(num_objects):
            mesh = trimesh.creation.box(extents=(np.random.uniform(0.1, 1.0),
                                                 np.random.uniform(0.1, 1.0),
                                                 np.random.uniform(0.1, 1.0)))
            mesh_list.append(mesh)

        # randomly sample points
        points = list()
        for mesh in mesh_list:
            pts, _ = trimesh.sample.sample_surface(mesh, count=num_all_points)
            points.append(pts)
        points = np.stack(points, axis=0) # (num_objects, 65536, 3)
        points = torch.from_numpy(points).float()

        # show whether consistent
        # NOTE their hash codes should be the same
        logger_info(f"hash code of the whole dataset = {get_tensor_hash(points)}", collective=True)

        # pick sub-dataset
        num_objects_per_process = num_objects // num_rank
        points = points[rank * num_objects_per_process : (rank + 1) * num_objects_per_process]

        # save
        self.points = points.to(device)
        self.batch_size = batch_size
        self.num_objects = num_objects
        self.num_points = num_points
        self.num_all_points = num_all_points
        self.num_objects_per_process = num_objects_per_process

    def __len__(self):
        return self.num_objects_per_process
    
    def __getitem__(self, _):
        # random sample
        ind_pts = torch.multinomial(torch.ones(self.num_all_points, device=device), self.num_points)
        ind_bs = torch.multinomial(torch.ones(self.num_objects_per_process, device=device), self.batch_size)
        points = self.points[ind_bs][:, ind_pts]
        # do augmentation
        rotation = Rotation.random(self.batch_size).as_matrix() # (batch_size, 3, 3)
        rotation = torch.from_numpy(rotation).to(device).float()
        translation = torch.randn([self.batch_size, 1, 1], device=device).mul(0.5)
        points = points @ rotation + translation
        # return
        return dict(points=points,
                    labels=ind_bs + rank * self.num_objects_per_process,
                )

# # # # # # # # # # # # # # # # 

loader_config = dict(
    batch_size=None,
    num_workers=0,
    pin_memory=False,
)

dataset = PrimitiveDataset(batch_size=BATCH_SIZE // num_rank, # one process batch size
                           num_objects=NUM_OBJECTS, # all processes num object
                           num_points=NUM_INPUT_POINTS, 
                           num_all_points=65536)
dataloaders = dict(train = torch.utils.data.DataLoader(dataset, shuffle=True, **loader_config),
                   val   = torch.utils.data.DataLoader(dataset, shuffle=False, **loader_config),
                )

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

model = torch.nn.Sequential( # NOTE a simple pointnet model
    torch.nn.Conv1d(3, 512, kernel_size=1), torch.nn.BatchNorm1d(512), torch.nn.ReLU(inplace=True),
    torch.nn.Conv1d(512, 512, kernel_size=1), torch.nn.BatchNorm1d(512), torch.nn.ReLU(inplace=True),
    torch.nn.Conv1d(512, 512, kernel_size=1), torch.nn.BatchNorm1d(512), torch.nn.ReLU(inplace=True),
    torch.nn.Conv1d(512, 512, kernel_size=1), torch.nn.BatchNorm1d(512), torch.nn.ReLU(inplace=True),
    torch.nn.Conv1d(512, 512, kernel_size=1), torch.nn.BatchNorm1d(512), torch.nn.ReLU(inplace=True),
    torch.nn.Conv1d(512, NUM_OBJECTS, kernel_size=1),
    torch.nn.AdaptiveMaxPool1d(output_size=1),
)
if num_rank > 1:
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
loss_fn = torch.nn.CrossEntropyLoss()

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.2, patience=1000)

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

def train_step_fn(batch_data):
    model.train()

    points = batch_data["points"].to(f"cuda:{rank}") # (batch_size, num_points, 3)
    labels = batch_data["labels"].to(f"cuda:{rank}")
    points = points.transpose(1, 2) # (batch_size, num_channels, num_points)
    pred = model(points)
    pred = pred.squeeze(2) # (batch_size, num_channels)
    loss = loss_fn(pred, labels)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    loss_dict = dict(loss=loss.item(),
                     lr=optimizer.param_groups[0]["lr"])
    
    return loss_dict

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

