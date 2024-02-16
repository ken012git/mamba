# build-in
import os
import json
import numpy as np
# torch
import torch
import torch.nn as nn

import torch
from mamba_ssm import Mamba
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel


class ForwardHook():
    def __init__(self, output_dir):
        self.output_dir = output_dir
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)

        self.step = 0

    def hook(self, m, x, y):
        print(f"forward {self.step}: ", m, x[0].shape, y[0].shape)
        if self.step == 0:
            # print(f"forward {self.step}: ", m, x[0].shape, y[0].shape)
            meta = {
                "module": str(m), 
                "input_shape": x[0].shape, 
                "output_shape": y[0].shape, 
            }
            if hasattr(m, "weight"):
                meta["weight"] = m.weight.shape
            
            if hasattr(m, "bias") and m.bias is not None:
                meta["bias"] = m.bias.shape

            with open(os.path.join(self.output_dir, f"meta.json"), 'w') as f:
                json.dump(meta, f, indent=4)

        self.step += 1
        x_np = x[0].clone().cpu().detach().numpy()
        y_np = y[0].clone().cpu().detach().numpy()
        with open(os.path.join(self.output_dir, f"input-{self.step}.npy"), 'wb') as f:
            np.save(f, x_np)
        with open(os.path.join(self.output_dir, f"output-{self.step}.npy"), 'wb') as f:
            np.save(f, y_np)



# class BackwardHook():
#     def __init__(self, output_dir):
#         self.output_dir = output_dir
#         if not os.path.isdir(output_dir):
#             os.makedirs(output_dir)

#         self.step = 0

#     def hook(self, m, gx, gy):
#         if self.step == 0:
#             print(f"backward {self.step}: ", m, gx[0].shape, gy[0].shape)
#             meta = {
#                 "module": str(m), 
#                 "grad_input_shape": gx[0].shape, 
#                 "grad_output_shape": gy[0].shape, 
#             }
#             with open(os.path.join(self.output_dir, f"meta.json"), 'w') as f:
#                 json.dump(meta, f, indent=4)

#         self.step += 1
#         gx_np = gx[0].clone().cpu().detach().numpy()
#         gy_np = gx[0].clone().cpu().detach().numpy()
#         with open(os.path.join(self.output_dir, f"grad_input-{self.step}.npy"), 'wb') as f:
#             np.save(f, gx_np)
#         with open(os.path.join(self.output_dir, f"grad_output-{self.step}.npy"), 'wb') as f:
#             np.save(f, gy_np)


# fwd_handler_list = []
            
def store_param(m, output_dir):

    if hasattr(m, "weight"):
        weight_np = m.weight.clone().cpu().detach().numpy()
        with open(os.path.join(output_dir, f"weight.npy"), 'wb') as f:
            np.save(f, weight_np)
    
    if hasattr(m, "bias") and m.bias is not None:
        bias_np = m.bias.clone().cpu().detach().numpy()
        with open(os.path.join(output_dir, f"bias.npy"), 'wb') as f:
            np.save(f, bias_np)

def add_hooks(model, output_dir):

    idx = 0
    for module in model.modules():
        if isinstance(module, nn.Embedding):
            print(idx, module)
            fwd_hook = ForwardHook(os.path.join(output_dir, "fwd", f"embed"))
            fwd_handler = module.register_forward_hook(fwd_hook.hook)
            # fwd_handler_list.append(fwd_handler)           
        if isinstance(module, Mamba):
            idx += 1
            print(idx, module)
            for name, mamba_module in module.named_modules():
                print(name, mamba_module)
                if hasattr(mamba_module, "weight"):
                    print("weight: ", mamba_module.weight.shape)
                if hasattr(mamba_module, "bias") and mamba_module.bias is not None:
                    print("bias: ", mamba_module.bias.shape)
                fwd_hook = ForwardHook(os.path.join(output_dir, "fwd", f"mamba_{idx}", f"{name}"))
                fwd_handler = mamba_module.register_forward_hook(fwd_hook.hook)
                store_param(mamba_module, os.path.join(output_dir, "fwd", f"mamba_{idx}", f"{name}"))
                # fwd_handler_list.append(fwd_handler)

            A_log = module.A_log.clone().cpu().detach().numpy()
            with open(os.path.join(os.path.join(output_dir, "fwd", f"mamba_{idx}"), f"A_log.npy"), 'wb') as f:
                np.save(f, A_log)
            D = module.D.clone().cpu().detach().numpy()
            with open(os.path.join(os.path.join(output_dir, "fwd", f"mamba_{idx}"), f"D.npy"), 'wb') as f:
                np.save(f, D)


# device = "cuda"
# dtype = torch.float16
# pretrained = "state-spaces/mamba-130m"
# model = MambaLMHeadModel.from_pretrained(pretrained, device=device, dtype=dtype)
# print(model)

# # state_dict = model.state_dict()
# # for k, v in state_dict.items():
# #     print(k, v.shape)

# add_hooks(model, "tmp")