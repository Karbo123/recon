import torch
import numpy as np
from collections import defaultdict


class GradClipper:
    """ adaptively clip nan, inf and norm
    """
    def __init__(self,
                 thres=0.75, # allows 25% bad gradients
                 len_buffer_min=1000,
                 len_buffer_max=10000,
                ):
        assert 0 <= thres <= 1
        self.thres = round(thres * 100)
        self.len_buffer_min = len_buffer_min
        self.len_buffer_max = len_buffer_max
        self.named_params_norm = defaultdict(list)


    @torch.no_grad()
    def __call__(self, module):
        assert isinstance(module, torch.nn.Module)
        
        for name, param in module.named_parameters():
            if param.grad is None: continue
            
            """ clip nan and inf """
            grad = torch.nan_to_num(param.grad.data, nan=0.0, posinf=0.0, neginf=0.0)

            """ compute new norms """
            norm = grad.norm()
            if len(self.named_params_norm[name]) > 0:
                """ compute ratio to clip """
                norm_clip = np.percentile(self.named_params_norm[name], self.thres)
                norm_clip_ratio = norm_clip / (norm + 1e-8)
            else:
                norm_clip_ratio = 1.0
            
            """ clip grad by norm """
            param.grad.data.copy_(grad.mul(norm_clip_ratio) if norm_clip_ratio < 1.0 else grad)

            """ save norms """
            self.named_params_norm[name].append(norm.item())
            if len(self.named_params_norm[name]) > self.len_buffer_max:
                self.named_params_norm[name] = self.named_params_norm[name][-self.len_buffer_min:]



if __name__ == "__main__":
    """ a simple example
    """
    model = torch.nn.Linear(3, 5)
    x = torch.randn(16, 3)
    x[torch.randint(16, size=(5, )), 0] = float("nan")
    x[torch.randint(16, size=(5, )), 0] = float("inf")
    x[torch.randint(16, size=(5, )), 2] = float("inf") * (-1)
    y = model(x)
    loss = y.sum()
    loss.backward()

    print("before clipping:")
    print(f"model.weight.grad = {model.weight.grad}")
    print(f"model.bias.grad = {model.bias.grad}")

    grad_clipper = GradClipper()
    grad_clipper(model)

    print("after clipping:")
    print(f"model.weight.grad = {model.weight.grad}")
    print(f"model.bias.grad = {model.bias.grad}")

