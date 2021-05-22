
import torch
from torch.autograd import Variable
from torch.nn.parallel._functions import Scatter, Gather
from torch_geometric.nn import DataParallel as TorchGeometricDataParallel_raw
from torch.nn import DataParallel as TorchDataParallel_raw


# def gather(outputs, target_device, dim=0):
#     def gather_map(outputs):
#         print(f"outputs = {outputs}")
#         out = outputs[0]
#         if "__stop" in out: # NOTE flag "__stop" tells to stop recursion
#             ############################
#             # remove flag
#             def remove_flag(pack):
#                 elem = pack[0]
#                 if not isinstance(elem, (tuple, list)):
#                     pack = type(pack)(x for x in pack if x not in ["__stop"])
#                     if len(pack): return pack
#                     else: return None
#                 pack = type(elem)(map(remove_flag, zip(*pack)))
#                 pack = type(elem)(x for x in pack if x is not None)
#                 return pack
#             try:
#                 outputs = remove_flag(outputs)
#             finally:
#                 remove_flag = None

#             ############################
#             # recursively move device
#             def move_device(pack):
#                 elem = pack[0]
#                 if isinstance(elem, Variable):
#                     for x in pack:
#                         print(type(x))
#                     print(f"==============")
#                     return type(pack)(x.to(device=target_device) for x in pack)
#                 return type(elem)(map(move_device, zip(*pack)))
#             try:
#                 outputs = move_device(outputs)
#             finally:
#                 move_device = None
            
#             ############################
#             # return
#             return outputs

#         elif isinstance(out, Variable):
#             return Gather.apply(target_device, dim, *outputs)
#         if out is None:
#             return None
#         return type(out)(map(gather_map, zip(*outputs)))

#     try:
#         return gather_map(outputs)
#     finally:
#         gather_map = None

def gather(outputs, target_device, dim=0, stop_keys=[]): # NOTE output of each device must be dict

    # out = outputs[0]
    # keys = list(out.keys())
    # results = dict()
    # for k in keys:
    #     if k not in stop_keys:
            
    #         def gather_map(outputs):
    #             out = outputs[0]
    #             if isinstance(out, Variable):
    #                 return Gather.apply(target_device, dim, *outputs)
    #             if out is None:
    #                 return None
    #             return type(out)(map(gather_map, zip(*outputs)))

    #         try:
    #             results[k] = gather_map([d[k] for d in outputs])
    #         finally:
    #             gather_map = None
        
    #     else:
    #         results[k] = [Gather.apply(target_device, dim, d[k]) for d in outputs]

    # return results

    raise # this has bug




class TorchGeometricDataParallel(TorchGeometricDataParallel_raw):
    def __init__(self, *args, **kwargs):
        self.stop_keys = kwargs.get("stop_keys", [])
        kwargs.pop("stop_keys")
        super().__init__(*args, **kwargs)
        
    def gather(self, outputs, output_device):
        return gather(outputs, output_device, dim=self.dim, stop_keys=self.stop_keys)


class TorchDataParallel(TorchDataParallel_raw):
    def __init__(self, *args, **kwargs):
        self.stop_keys = kwargs.get("stop_keys", [])
        kwargs.pop("stop_keys")
        super().__init__(*args, **kwargs)
        
    def gather(self, outputs, output_device):
        return gather(outputs, output_device, dim=self.dim, stop_keys=self.stop_keys)

