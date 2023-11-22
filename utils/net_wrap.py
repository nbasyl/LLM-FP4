import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.models import MatMul
import re

def wrap_modules_in_net(net,cfg):
    wrapped_modules={}
    module_dict={}
    module_types = {"embedding":"qembedding","q_proj":"qlinear_query", "k_proj":"qlinear_key", "v_proj":"qlinear_value",
                    "o_proj":'qlinear_o',
                    "gate_proj":'qlinear_gate',
                    "down_proj":'qlinear_down',
                    "up_proj":'qlinear_up',
                    'lm_head':'qlinear_score','matmul1':"qmatmul_qk", 'matmul2':"qmatmul_scorev"}
    
    device_map = net.hf_device_map
    
    it=[(name,m) for name,m in net.named_modules()]
    for name,m in it:
        module_dict[name]=m
        idx=name.rfind('.')
        if idx==-1:
            idx=0
        father_name=name[:idx]
        if father_name in module_dict:
            father_module=module_dict[father_name]
        else:
            raise RuntimeError(f"father module {father_name} not found")
        if isinstance(m,nn.Embedding):
            new_m = cfg.get_module("qembedding",m.num_embeddings,m.embedding_dim)
            new_m.weight.data=m.weight.data
            replace_m=new_m
            wrapped_modules[name] = new_m
            setattr(father_module,name[idx+1:],replace_m)
        elif isinstance(m,nn.Linear):
            if 'lm_head' in name:
                new_m = cfg.get_module(module_types[name[idx:]],m.in_features,m.out_features)
                new_m.weight.data=m.weight.data
                new_m.bias=m.bias
                replace_m=new_m
                wrapped_modules[name] = new_m
                setattr(father_module,name[idx:],replace_m)
            else:
                new_m = cfg.get_module(module_types[name[idx+1:]],m.in_features,m.out_features)
                new_m.weight.data=m.weight.data
                new_m.bias=m.bias
                replace_m=new_m
                wrapped_modules[name] = new_m
                setattr(father_module,name[idx+1:],replace_m)
        elif isinstance(m,MatMul):
            # print(f'name: {name[idx+1:]}')
            # Matmul Layer

            if name[:15][-1] == ".":
                layer_idx = name[:14]
            else:
                layer_idx =  name[:15]

            new_m = cfg.get_module(module_types[name[idx+1:]],device_map = f'cuda:{device_map[layer_idx]}')
            replace_m=new_m
            wrapped_modules[name] = new_m
            setattr(father_module,name[idx+1:],replace_m)
            
    print("Completed net wrap.")
    return wrapped_modules


