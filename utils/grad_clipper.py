import torch
from copy import deepcopy
from collections import defaultdict

# ref: https://github.com/aliutkus/torchpercentile/blob/master/torchpercentile/percentile.py
class Percentile(torch.autograd.Function):
    """
    torch Percentile autograd Functions subclassing torch.autograd.Function
    computes the percentiles of a tensor over the first axis
    """
    def __call__(self, input, percentiles):
        return self.forward(input, percentiles)

    # def __init__(self, nb_percentiles):
    #     """ Inits a Percentile object with `nb_percentiles` regularly spaced
    #     between 0 and 100"""
    #     self.nb_percentiles = nb_percentiles
    #     self.percentiles = torch.linspace(0, 100, nb_percentiles)

    def forward(self, input, percentiles):
        """
        Find the percentiles of a tensor along the first dimension.
        """
        input_dtype = input.dtype
        input_shape = input.shape
        if not isinstance(percentiles, torch.Tensor):
            percentiles = torch.tensor(percentiles, dtype=torch.double)
        if not isinstance(percentiles, torch.Tensor):
            percentiles = torch.tensor(percentiles)
        input = input.double()
        percentiles = percentiles.to(input.device).double()
        input = input.view(input.shape[0], -1)
        in_sorted, in_argsort = torch.sort(input, dim=0)
        positions = percentiles * (input.shape[0]-1) / 100
        floored = torch.floor(positions)
        ceiled = floored + 1
        ceiled[ceiled > input.shape[0] - 1] = input.shape[0] - 1
        weight_ceiled = positions-floored
        weight_floored = 1.0 - weight_ceiled
        d0 = in_sorted[floored.long(), :] * weight_floored[:, None]
        d1 = in_sorted[ceiled.long(), :] * weight_ceiled[:, None]
        self.save_for_backward(input_shape, in_argsort, floored.long(),
                               ceiled.long(), weight_floored, weight_ceiled)
        result = (d0+d1).view(-1, *input_shape[1:])
        return result.type(input_dtype)

    def backward(self, grad_output):
        """
        backward the gradient is basically a lookup table, but with weights
        depending on the distance between each point and the closest
        percentiles
        """
        (input_shape, in_argsort, floored, ceiled,
         weight_floored, weight_ceiled) = self.saved_tensors

        # the argsort in the flattened in vector

        cols_offsets = (
            torch.arange(
                    0, input_shape[1], device=in_argsort.device)
            )[None, :].long()
        in_argsort = (in_argsort*input_shape[1] + cols_offsets).view(-1).long()
        floored = (
            floored[:, None]*input_shape[1] + cols_offsets).view(-1).long()
        ceiled = (
            ceiled[:, None]*input_shape[1] + cols_offsets).view(-1).long()

        grad_input = torch.zeros((in_argsort.size()), device=self.device)
        grad_input[in_argsort[floored]] += (grad_output
                                            * weight_floored[:, None]).view(-1)
        grad_input[in_argsort[ceiled]] += (grad_output
                                           * weight_ceiled[:, None]).view(-1)

        grad_input = grad_input.view(*input_shape)
        return grad_input



class GradClipper:
    """ adaptively clip nan, inf and norm
    """
    def __init__(self,
                 thres=0.75, # allows 25% bad gradients
                 max_buffer=10000,
                 retain=0.9,
                 device="cpu",
                ):
        assert 0 <= thres <= 1
        self.thres = round(thres * 100)
        self.max_buffer = max_buffer
        self.retain = retain
        self.buffer = defaultdict(lambda: torch.zeros([max_buffer], device=device))
        self.buffer_len = defaultdict(int)
        self.percentile = Percentile()

    @torch.no_grad()
    def __call__(self, module):
        assert isinstance(module, torch.nn.Module)
        
        for name, param in module.named_parameters():
            if param.grad is None: continue
            
            """ clip nan and inf """
            grad = torch.nan_to_num(param.grad.data, nan=0.0, posinf=0.0, neginf=0.0)

            """ compute new norms """
            norm = grad.norm()
            if self.buffer_len[name] > 0:
                """ compute ratio to clip """
                norm_clip = self.percentile(self.buffer[name][:self.buffer_len[name]], [self.thres])
                norm_clip_ratio = norm_clip / norm.clamp(min=1e-7)
            else:
                norm_clip_ratio = 1.0
            
            """ clip grad by norm """
            param.grad.data.copy_(grad.mul(norm_clip_ratio) if norm_clip_ratio < 1.0 else grad)

            """ save norms """
            if self.buffer_len[name] >= self.max_buffer:
                cut = round(self.retain * self.max_buffer)
                self.buffer[name][:cut] = self.buffer[name][-cut:].clone()
                self.buffer_len[name] = cut
            self.buffer[name][self.buffer_len[name]] = norm
            self.buffer_len[name] += 1

    # grad clipper also has state_dict, help to exactly resume training
    def state_dict(self):
        return dict(thres=self.thres, max_buffer=self.max_buffer, retain=self.retain,
                    buffer=self.buffer,
                    buffer_len=self.buffer_len,
                    )
    
    def load_state_dict(self, sd):
        self.thres = sd["thres"]
        self.max_buffer = sd["max_buffer"]
        self.retain = sd["retain"]
        self.buffer = deepcopy(sd["buffer"])
        self.buffer_len = deepcopy(sd["buffer_len"])


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

