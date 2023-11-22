from numpy import isin
import torch
from quant_layers.fp_linear import FPMinMaxQuantLinear
from quant_layers.fp_embed import FPMinMaxQuantEmbedding
import torch.nn.functional as F
from tqdm import tqdm
import os

class QuantCalibrator():
    """
    Modularization of quant calib.

    Notice: 
    all quant modules has method "calibration_step1" that should only store raw inputs and outputs
    all quant modules has method "calibration_step2" that should only quantize its intervals
    and we assume we could feed in all calibration data in one batch, without backward propagations

    sequential calibration is memory-friendly, while parallel calibration may consume 
    hundreds of GB of memory.
    """
    def __init__(self, net, wrapped_modules, calib_loader, sequential=True):
        self.net = net
        self.wrapped_modules = wrapped_modules
        self.calib_loader = calib_loader
        self.sequential = sequential
        self.calibrated = False
    
    def sequential_quant_calib(self):
        """
        A quick implementation of calibration.
        Assume calibration dataset could be fed at once.
        """
        # run calibration
        n_calibration_steps=2
        for step in range(n_calibration_steps):
            print(f"Start calibration step={step+1}")
            for name,module in self.wrapped_modules.items():
                # corner cases for calibrated modules
                if hasattr(module, "calibrated"):
                    if step == 1:
                        module.mode = "raw"
                    elif step == 2:
                        module.mode = "quant_forward"
                else:
                    module.mode=f'calibration_step{step+1}'
            with torch.no_grad():
                for inp,target in self.calib_loader:
                    inp=inp.cuda()
                    self.net(inp)
        
        # finish calibration
        for name,module in self.wrapped_modules.items():
            module.mode='quant_forward'
        torch.cuda.empty_cache() # memory footprint cleanup
        print("sequential calibration finished")
    
    def parallel_quant_calib(self):
        """
        A quick implementation of parallel quant calib
        Assume calibration dataset could be fed at once, and memory could hold all raw inputs/outs
        """
        # calibration step1: collect raw data
        print(f"Start calibration step=1")
        for name,module in self.wrapped_modules.items():
            # corner cases for calibrated modules
            if hasattr(module, "calibrated"):
                module.mode = "raw"
            else:
                module.mode=f'calibration_step1'
        with torch.no_grad():
            for inp,target in self.calib_loader:
                inp=inp.cuda()
                self.net(inp)
        # calibration step2: each module run calibration with collected raw data
        for name,module in self.wrapped_modules.items():
            if hasattr(module, "calibrated"):
                continue
            else:
                module.mode=f"calibration_step2"
                with torch.no_grad():
                    if isinstance(module, FPMinMaxQuantLinear):
                        module.forward(module.raw_input.cuda())
                    elif isinstance(module, FPMinMaxQuantMatMul):
                        module.forward(module.raw_input[0].cuda(), module.raw_input[1].cuda())
                    torch.cuda.empty_cache()
                
        # finish calibration
        for name,module in self.wrapped_modules.items():
            module.mode='quant_forward'
        torch.cuda.empty_cache() # memory footprint cleanup
        print("calibration finished")
    
    def quant_calib(self):
        calib_layers=[]
        for name,module in self.wrapped_modules.items():
            calib_layers.append(name)
        print(f"prepare parallel calibration for {calib_layers}")
        if self.sequential:
            self.sequential_quant_calib()
        else:
            self.parallel_quant_calib()
        self.calibrated = True

    def batching_quant_calib(self):
        calib_layers=[]
        for name,module in self.wrapped_modules.items():
            calib_layers.append(name)
        print(f"prepare parallel calibration for {calib_layers}")

        print("start calibration")

        # assume wrapped modules are in order (true for dict in python>=3.5)
        q = tqdm(self.wrapped_modules.items(), desc="Brecq")
        for name, module in q:
            q.set_postfix_str(name)

            # add fp and bp hooks to current modules, which bypass calibration step 1
            # precedent modules are using quant forward
            hooks = []
            if isinstance(module, FPMinMaxQuantLinear):
                hooks.append(module.register_forward_hook(linear_forward_hook))
            if isinstance(module, FPMinMaxQuantMatMul):
                hooks.append(module.register_forward_hook(matmul_forward_hook))
            
            # feed in calibration data, and store the data
            for inp, target in self.calib_loader:
                for batch_st in range(0,self.calib_loader.batch_size,self.batch_size):
                    self.net.zero_grad()
                    inp_ = inp[batch_st:batch_st+self.batch_size].cuda()
                    self.net(inp_)
                del inp, target
                torch.cuda.empty_cache()
            
            # replace cached raw_inputs, raw_outs
            if isinstance(module, FPMinMaxQuantLinear):
                module.raw_input = torch.cat(module.raw_input, dim=0)
                module.raw_out = torch.cat(module.raw_out, dim=0)
    
            if isinstance(module, FPMinMaxQuantMatMul):
                module.raw_input = [torch.cat(_, dim=0) for _ in module.raw_input]
                module.raw_out = torch.cat(module.raw_out, dim=0)
            for hook in hooks:
                hook.remove()

            # run calibration step2
            with torch.no_grad():
                if isinstance(module, FPMinMaxQuantLinear):
                    module.calibration_step2()

                if isinstance(module, FPMinMaxQuantMatMul):
                    module.calibration_step2()
                torch.cuda.empty_cache()
            
            # finishing up current module calibration
            if self.sequential:
                module.mode = "quant_forward"
            else:
                module.mode = "raw"

        # finish calibration
        for name, module in self.wrapped_modules.items():
            module.mode = "quant_forward"
        
        print("calibration finished")

def linear_forward_hook(module, input, output):
    if module.raw_input is None:
        module.raw_input = []
    if module.raw_out is None:
        module.raw_out = []

    module.raw_input.append(input[0].cpu().detach())
    module.raw_out.append(output.cpu().detach())

def matmul_forward_hook(module, input, output):
    if module.raw_input is None:
        module.raw_input = [[],[]]
    if module.raw_out is None:
        module.raw_out = []
    module.raw_input[0].append(input[0].cpu().detach())
    module.raw_input[1].append(input[1].cpu().detach())
    module.raw_out.append(output.cpu().detach())

class FloatingPointQuantCalibrator(QuantCalibrator):
    """
    MSE metric needs layer outputs to weigh the loss, both sequentially
    and parallelly.
    """
    def __init__(self, net, wrapped_modules, calib_loader, sequential=False, batch_size=1):
        super().__init__(net, wrapped_modules, calib_loader, sequential=sequential)
        self.batch_size = batch_size

    def batching_quant_calib(self):
        self.net.train()
        calib_layers=[]
        for name,module in self.wrapped_modules.items():
            calib_layers.append(name)
        print(f"prepare parallel calibration for {calib_layers}")
        print(f"self.net {self.net.training}")
        print("start calibration for fp")

        # assume wrapped modules are in order (true for dict in python>=3.5)
        q = tqdm(self.wrapped_modules.items())
        for name, module in q:
            q.set_postfix_str(name)
            print(f"start quantizing layer {name}")
            # add fp and bp hooks to current modules, which bypass calibration step 1
            # precedent modules are using quant forward
            hooks = []
            if isinstance(module, FPMinMaxQuantLinear):
                hooks.append(module.register_forward_hook(linear_forward_hook))

            # feed in calibration data, and store the data
            idx = 0
            for batch_ in self.calib_loader:
                for batch_st in range(0,self.calib_loader.batch_size,self.batch_size):
                    batch_ = tuple(t for t in batch_)
                    self.net.zero_grad()
                    
                    input_ids, label_ids = batch_
                    input_ids_ = input_ids[batch_st:batch_st+self.batch_size].cuda()
                    model_output = self.net(input_ids_[0])
                    pred = model_output.logits
                    idx += 1
                    del input_ids_
                    
                del pred
                torch.cuda.empty_cache()
            
            # replace cached raw_inputs, raw_outs
            if isinstance(module, FPMinMaxQuantLinear):
                module.raw_input = torch.cat(module.raw_input, dim=0)
                module.raw_out = torch.cat(module.raw_out, dim=0)
            if hasattr(module, "metric") and module.metric != None and isinstance(module, FPMinMaxQuantEmbedding) == False:
                module.raw_grad = None
            for hook in hooks:
                hook.remove()

            # run calibration step2
            with torch.no_grad():
                if isinstance(module, FPMinMaxQuantEmbedding):
                    module.calibration_step2()
                if isinstance(module, FPMinMaxQuantLinear):
                    module.calibration_step2()
                torch.cuda.empty_cache()
            
            # finishing up current module calibration
            if self.sequential:
                module.mode = "quant_forward"
            else:
                module.mode = "raw"

        # finish calibration
        for name, module in self.wrapped_modules.items():
            module.mode = "quant_forward"

        print("L2_norm calibration finished")
    
    def save_qparams(self, config_name, save_name):
        state_dict = self.net.state_dict()
        to_save = {}
        if config_name == 'FPQ_config_llama':
            q_ptq_params = ['exponent_bit','mantissa_bit','interval','a_bias']
        else:
            q_ptq_params = ['exponent_bit','mantissa_bit','interval']
        for key in state_dict.keys():
            if any(t in key for t in q_ptq_params):
                to_save[key] = state_dict[key]

        save_directory = f"./search_result/{config_name}/"
        os.makedirs(save_directory, exist_ok=True)
        torch.save(to_save, os.path.join(save_directory,f"{save_name}.pt"))
        print("Successfully saved the searched result")
                
class FloatingPointQuantInitialization(QuantCalibrator):
    """
    MSE metric needs layer outputs to weigh the loss, both sequentially
    and parallelly.
    """
    def __init__(self, net, wrapped_modules, calib_loader, sequential=False, batch_size=1):
        super().__init__(net, wrapped_modules, calib_loader, sequential=sequential)
        self.batch_size = batch_size

    def batching_quant_init(self):
        self.net.train()
        calib_layers=[]
        for name,module in self.wrapped_modules.items():
            calib_layers.append(name)
        print("start initialization for FP PTQ")

        # assume wrapped modules are in order (true for dict in python>=3.5)
        q = tqdm(self.wrapped_modules.items())
        for name, module in q:
            q.set_postfix_str(name)
            print(f"start initilizaing layer {name}")
            # add fp and bp hooks to current modules, which bypass calibration step 1
            # precedent modules are using quant forward
            hooks = []
            if isinstance(module, FPMinMaxQuantLinear):
                hooks.append(module.register_forward_hook(linear_forward_hook))

            # feed in calibration data, and store the data
            idx = 0
            for batch_ in self.calib_loader:
                for batch_st in range(0,self.calib_loader.batch_size,self.batch_size):
                    batch_ = tuple(t for t in batch_)
                    self.net.zero_grad()
                    
                    input_ids, label_ids = batch_
                    input_ids_ = input_ids[batch_st:batch_st+self.batch_size].cuda()
                    model_output = self.net(input_ids_[0])
                    pred = model_output.logits
                    idx += 1
                    del input_ids_
                    
                del pred
                torch.cuda.empty_cache()
            
            # replace cached raw_inputs, raw_outs
            if isinstance(module, FPMinMaxQuantLinear):
                module.raw_input = torch.cat(module.raw_input, dim=0)
                module.raw_out = torch.cat(module.raw_out, dim=0)
            if hasattr(module, "metric") and module.metric != None and isinstance(module, FPMinMaxQuantEmbedding) == False:
                module.raw_grad = None
            for hook in hooks:
                hook.remove()

            with torch.no_grad():
                if isinstance(module, FPMinMaxQuantEmbedding):
                    module._initialize_intervals_eval()
                if isinstance(module, FPMinMaxQuantLinear):
                    module._initialize_intervals_eval()
                torch.cuda.empty_cache()
            
            if self.sequential:
                module.mode = "quant_forward"
            else:
                module.mode = "raw"

        for name, module in self.wrapped_modules.items():
            module.mode = "quant_forward"

        print("L2_norm calibration finished")

