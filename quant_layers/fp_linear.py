
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import math

class FPMinMaxQuantLinear(nn.Linear):
    def __init__(self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        mode = "raw",
        w_bit = 8,
        a_bit = 8,
        bias_bit = None,
        bias_correction=False,
        w_exponent_bit = 4, a_exponent_bit = 4):
        super().__init__(in_features,out_features,bias)
        self.n_calibration_step=2
        self.mode = mode
        self.w_bit = w_bit
        self.a_bit = a_bit
        self.bias_bit=bias_bit
        assert bias_bit is None,"No support bias bit now"
        
        assert w_exponent_bit < (w_bit - 1)
        assert a_exponent_bit < (a_bit - 1)

        self.register_buffer('w_exponent_bit',torch.tensor(w_exponent_bit))
        self.register_buffer('w_mantissa_bit',torch.tensor(w_bit - 1 - w_exponent_bit))

        self.register_buffer('a_exponent_bit',torch.tensor(a_exponent_bit))
        self.register_buffer('a_mantissa_bit',torch.tensor(a_bit - 1 - a_exponent_bit))

        self.register_buffer('w_interval',None)
        self.register_buffer('a_interval',None)

        self.raw_input=None
        self.raw_out=None
        self.metric=None
        self.bias_correction = bias_correction
        self.default_bias = 2 ** (self.w_exponent_bit - 1)

        
    def get_maxval_from_bias(self, act_or_weight):
        if act_or_weight == 0 and self.a_interval != None:
            return (2 - 2 ** (-self.a_mantissa_bit)) * 2 ** (
                2**self.a_exponent_bit - 1 - self.a_interval
            )
        elif act_or_weight == 1 and self.w_interval != None:
            return (2 - 2 ** (-self.w_mantissa_bit)) * 2 ** (
                2**self.w_exponent_bit - 1 - self.w_interval
            )
        else:
            raise AssertionError
    
    def get_log_scale(self, x ,act_or_weight):
        
        if act_or_weight == 0:
            a_maxval = self.get_maxval_from_bias(0)
            a_bias = self.a_interval if self.a_interval != None else self.default_bias
            a_bias = a_bias.float()
            a_minval = -a_maxval
            a = torch.min(torch.max(x, a_minval), a_maxval)
            a_log_scales = torch.clamp((torch.floor(torch.log2(torch.abs(a)) + a_bias)).detach(), 1.0)
            return a, 2.0 ** (a_log_scales - self.a_mantissa_bit - a_bias)
        
        elif act_or_weight == 1:
            w_maxval = self.get_maxval_from_bias(1)
            w_bias = self.w_interval if self.w_interval != None else self.default_bias
            w_bias = w_bias.float()
            w_minval = -w_maxval
            w = torch.min(torch.max(x, w_minval), w_maxval)
            w_log_scales = torch.clamp((torch.floor(torch.log2(torch.abs(w)) + w_bias)).detach(), 1.0)
            return w, 2.0 ** (w_log_scales - self.w_mantissa_bit - w_bias)

    def get_scale(self, input, bits, mantissa_bit, bias):
        
        M = mantissa_bit
        E = bits - 1 - M
        maxval = (2 - 2 ** (-M)) * 2 ** (
                2**E - 1 - bias
            )
        minval = -maxval
        input = torch.min(torch.max(input, minval), maxval)
        input_log_scales = torch.clamp((torch.floor(torch.log2(torch.abs(input)) + bias)).detach(), 1.0)
        return input, 2.0 ** (input_log_scales - M - bias.float())        


    def forward(self, x):
        if self.mode=='raw':
            out=F.linear(x, self.weight, self.bias)
        elif self.mode=="quant_forward":
            out=self.quant_forward(x)
        elif self.mode=="calibration_step1":
            out=self.calibration_step1(x)
        elif self.mode=="calibration_step2":
            out=self.calibration_step2(x)
        else:
            raise NotImplementedError
        return out
    
    def quant_weight_bias(self):
        w, w_scale = self.get_log_scale( self.weight ,act_or_weight = 1)
        w=(w/w_scale).round_()
        w_sim=w.mul_(w_scale)
        if self.bias is not None:
            return w_sim,self.bias
        else:
            return w_sim,None
    
    def quant_input(self, x):
        a, a_scale = self.get_log_scale( x ,act_or_weight = 0)
        x_sim=(a/a_scale).round_()
        x_sim.mul_(a_scale)
        return x_sim
    
    def quant_forward(self,x):
        assert self.calibrated is not None,f"You should run calibrate_forward before run quant_forward for {self}"
        w_sim,bias_sim=self.quant_weight_bias()
        x_sim=self.quant_input(x)
        out=F.linear(x_sim, w_sim, bias_sim)
        return out
    
    def _bias_correction_quant_forward(self, x):
        if self.bias_correction and self.bias != None:
            w_sim = self.quant_weight_bias()[0]
            x_sim = self.quant_input(x)
            eps = F.linear(x_sim, w_sim-self.weight.data, None)
            eps = torch.mean(eps, dim=(list(range(len(eps.shape)-1))), keepdim=False)
            self.bias -= eps
            self.bias_correction = False
        return self.quant_forward(x)

    def calibration_step1(self,x):
        # step1: collection the FP32 values
        out=F.linear(x, self.weight, self.bias)
        self.raw_input=x.cpu().detach()
        self.raw_out=out.cpu().detach()
        return out
    
    def calibration_step2(self,x):
        # step2: search for the best bias for w and a of each layer

        w_maxval = self.weight.data.abs().max()
        self.w_interval = 2**self.w_exponent_bit - torch.log2(w_maxval) + math.log2(2 - 2 ** (-self.w_mantissa_bit)) - 1
        

        x_maxval = x.abs().max()
        self.a_interval = 2**self.a_exponent_bit - torch.log2(x_maxval) + math.log2(2 - 2 ** (-self.a_mantissa_bit)) - 1

        self.calibrated=True
        out=self._bias_correction_quant_forward(x)
        return out

class FPPTQSLQuantLinear(FPMinMaxQuantLinear):
    """
    Floating Point PTQSL on linear modules.
    """
    def __init__(self, 
        in_features: int,
        out_features: int,
        bias: bool = True,
        mode = "raw",
        w_bit = 8,
        a_bit = 8,
        bias_bit = None,
        bias_correction = False,
        w_exponent_bit = 4, a_exponent_bit = 4,
        metric="L2_norm", search_round=1, eq_alpha=0, eq_beta=1, eq_n=100, parallel_eq_n=10, n_H=1, n_V=1, n_a=1):
        super().__init__(in_features, out_features, bias=bias, mode=mode, w_bit=w_bit, a_bit=a_bit, bias_bit=bias_bit, bias_correction=bias_correction, w_exponent_bit=w_exponent_bit, a_exponent_bit=a_exponent_bit)
        self.metric = metric
        self.search_round = search_round
        self.eq_alpha = eq_alpha
        self.eq_beta = eq_beta
        self.eq_n = int(eq_n)
        self.n_H = n_H
        self.n_V = n_V
        self.n_a = n_a
        self.crb_rows = out_features // n_V
        self.crb_cols = in_features // n_H # ignore remnent != 0 situations
        self.crb_acts = in_features // n_a
        self.parallel_eq_n = parallel_eq_n
        # self.init_layerwise = init_layerwise deprecated 
        self.raw_grad = None

    def _get_similarity(self, tensor_raw, tensor_sim, metric=None):
        """
        tensor_raw: *, features
        tensor_sim: *, features
        similarity: *
        It's your job to calculate mean on * dims!
        """
        if metric == "cosine":
            similarity = F.cosine_similarity(tensor_raw, tensor_sim, dim=-1)
        elif metric == "pearson":
            similarity = F.cosine_similarity(tensor_raw-torch.mean(tensor_raw,dim=-1,keepdim=True), tensor_sim-torch.mean(tensor_sim,dim=-1,keepdim=True), dim=-1)
        else:
            if metric == "L1_norm":
                similarity = -torch.abs(tensor_raw - tensor_sim)
            elif metric == "L2_norm":
                similarity = -(tensor_raw - tensor_sim) ** 2
            elif metric == "linear_weighted_L2_norm":
                similarity = -tensor_raw.abs() * (tensor_raw - tensor_sim) ** 2
            elif metric == "square_weighted_L2_norm":
                similarity = -(tensor_raw * (tensor_raw - tensor_sim)) ** 2
            else:
                raise NotImplementedError(f"metric {metric} not implemented!")
            similarity = torch.mean(similarity, dim=-1)
        return similarity
    
    def quant_weight_bias(self):
        # self.w_interval shape: n_V, 1, n_H, 1
        w_sim = self.weight.view(self.n_V,self.crb_rows,self.n_H,self.crb_cols)
        w, w_scale = self.get_log_scale(w_sim ,act_or_weight = 1)
        w_sim = (w / w_scale).round_()
        w_sim = w_sim.mul_(w_scale).view(self.out_features,self.in_features)
        if self.bias is not None:
            return w_sim,self.bias
        else:
            return w_sim,None
    
    def quant_input(self, x):
        # self.a_interval shape: n_a,1
        x_sim=torch.cat(torch.chunk(x.unsqueeze(-2), chunks=self.n_a, dim=-1), dim=-2)
        
        a, a_scale = self.get_log_scale(x_sim, act_or_weight = 0)

        x_sim=(a / a_scale).round_()
        x_sim = x_sim.mul_(a_scale).reshape_as(x)
        return x_sim

    def _search_best_w_interval(self, x, weight_interval_candidates, raw_out_expanded_chunked):
        """
        Modularization of searching best weight intervals
        """
        tmp_w_interval = self.w_interval.unsqueeze(0) # shape: 1,n_V,1,n_H,1
        for h in range(self.n_H):
            similarities = []
            for p_st in range(0,self.eq_n,self.parallel_eq_n):
                p_ed = min(self.eq_n, p_st+self.parallel_eq_n)
                cur_w_interval = tmp_w_interval.repeat(p_ed-p_st,1,1,1,1)
                cur_w_interval[:,:,:,h:h+1,:] = weight_interval_candidates[p_st:p_ed,:,:,h:h+1,:]
                # quantize weight and bias 
                w_sim = self.weight.view(self.n_V,self.crb_rows,self.n_H,self.crb_cols).unsqueeze(0) # shape: 1,n_V,crb_rows,n_H,crb_cols
                
                w, cur_w_scale = self.get_scale(w_sim, bits = self.w_bit, mantissa_bit= self.w_mantissa_bit, bias= cur_w_interval)

                w_sim = (w/cur_w_scale).round_().mul_(cur_w_scale) # shape: parallel_eq_n,n_V,crb_rows,n_H,crb_cols
                w_sim = w_sim.view(-1,self.in_features) # shape: parallel_eq_n*oc,ic
                bias_sim = self.bias.repeat(p_ed-p_st) if self.bias is not None else None
                # quantize input
                x_sim = self.quant_input(x)
                # calculate similarity and store them
                out_sim = F.linear(x_sim, w_sim, bias_sim) # shape: B,*,parallel_eq_n*oc
                out_sim = torch.cat(torch.chunk(out_sim.unsqueeze(-2), chunks=p_ed-p_st, dim=-1), dim=-2) # shape: B,*,parallel_eq_n,oc
                out_sim = torch.cat(torch.chunk(out_sim.unsqueeze(-2), chunks=self.n_V, dim=-1), dim=-2) # shape: B,*,parallel_eq_n,n_V,crb_rows
                similarity = self._get_similarity(raw_out_expanded_chunked, out_sim, self.metric) # shape: B,*,parallel_eq_n,n_V
                similarity = torch.mean(similarity, dim=list(range(len(similarity.shape)-2))) # shape: parallel_eq_n, n_V
                similarities.append(similarity)
            # store best weight interval of h into tmp_w_interval
            similarities = torch.cat(similarities, dim=0) # shape: eq_n, n_V
            h_best_index = similarities.argmax(dim=0).reshape(1,-1,1,1,1) # shape: 1,n_V,1,1,1
            tmp_w_interval[:,:,:,h:h+1,:] = torch.gather(weight_interval_candidates[:,:,:,h:h+1,:],dim=0,index=h_best_index)
        self.w_interval = tmp_w_interval.squeeze(dim=0)
    
    def _search_best_a_interval(self, x, input_interval_candidates, raw_out_expanded):
        tmp_a_interval = self.a_interval.unsqueeze(-1) # shape: n_a,1,1
        for a in range(self.n_a):
            similarities = []
            for p_st in range(0,self.eq_n,self.parallel_eq_n):
                p_ed = min(self.eq_n, p_st+self.parallel_eq_n)
                cur_a_interval = tmp_a_interval.repeat(1,1,p_ed-p_st) # shape: n_a,1,parallel_eq_n
                cur_a_interval[a:a+1,:,:] = input_interval_candidates[a:a+1,:,p_st:p_ed]
                # quantize weight and bias 
                w_sim, bias_sim = self.quant_weight_bias()
                # quantize input
                x_sim=torch.cat(torch.chunk(x.unsqueeze(-2), chunks=self.n_a, dim=-1), dim=-2).unsqueeze(-1)
                
                cur_a, cur_a_scale = self.get_scale(x_sim, bits = self.a_bit, mantissa_bit= self.a_mantissa_bit, bias= cur_a_interval)

                x_sim=(cur_a/(cur_a_scale)).round_()*(cur_a_scale) # shape: B,*,n_a,crb_acts,parallel_eq_n
                x_sim = x_sim.permute(*list(range(len(x_sim.shape)-3)),-1,-3,-2).reshape(*x.shape[:-1],p_ed-p_st,x.shape[-1]) # shape: B,*,parallel_eq_n,ic
                # calculate similarity and store them
                out_sim = F.linear(x_sim, w_sim, bias_sim) # shape: B,*,parallel_eq_n,oc
                similarity = self._get_similarity(raw_out_expanded, out_sim, self.metric) # shape: B,*,parallel_eq_n
                similarity = torch.mean(similarity, dim=list(range(len(similarity.shape)-1))) # shape: parallel_eq_n
                similarities.append(similarity)
            # store best input interval and store in tmp_a_interval
            similarities = torch.cat(similarities, dim=0) # shape: eq_n
            a_best_index = similarities.argmax(dim=0, keepdim=True).reshape(1,1,-1)
            tmp_a_interval[a:a+1,:,:] = torch.gather(input_interval_candidates[a:a+1,:,:],dim=2,index=a_best_index)
        self.a_interval = tmp_a_interval.squeeze(-1)

    def _initialize_intervals(self, x):
        
        x_maxval = x.abs().max()
        self.a_interval=(2**self.a_exponent_bit - torch.log2(x_maxval) + math.log2(2 - 2 ** (-self.a_mantissa_bit)) - 1).detach().view(1,1).repeat(self.n_a,1)


        self.n_V = self.out_features
        self.crb_rows = self.out_features // self.n_V
        w_maxval = self.weight.view(self.n_V,self.crb_rows,self.n_H,self.crb_cols).abs().amax([1,3],keepdim=True)
        self.w_interval=(2**self.w_exponent_bit - torch.log2(w_maxval) + math.log2(2 - 2 ** (-self.w_mantissa_bit)) - 1)
       
    def calibration_step2(self,x):
        # initialize intervals with minmax intervals
        self._initialize_intervals(x)

        # put raw outs on GPU
        raw_out_expanded = self.raw_out.to(x.device).unsqueeze(-2)  # shape: B,*,1,oc
        raw_out_expanded_chunked = torch.cat(torch.chunk(raw_out_expanded.unsqueeze(-2), chunks=self.n_V, dim=-1), dim=-2) # shape: B,*,1,n_V,crb_rows
        
        # put raw grad on GPU
        self.raw_grad = self.raw_grad.to(x.device) if self.raw_grad != None else None

        # prepare weight intervals and similarities
        weight_interval_candidates = torch.tensor([self.eq_alpha + i*(self.eq_beta - self.eq_alpha)/self.eq_n for i in range(self.eq_n + 1)]).to(self.weight.device).view(-1,1,1,1,1) * self.w_interval.unsqueeze(0) # shape: eq_n,n_V,1,n_H,1
        input_interval_candidates =  torch.tensor([self.eq_alpha + i*(self.eq_beta - self.eq_alpha)/self.eq_n for i in range(self.eq_n + 1)]).to(self.weight.device).view(1,1,-1) * self.a_interval.unsqueeze(-1) # shape: n_a,1,eq_n
        for e in range(self.search_round):
            # search for best weight interval
            self._search_best_w_interval(x, weight_interval_candidates, raw_out_expanded_chunked)
            # search for best input interval
            self._search_best_a_interval(x, input_interval_candidates, raw_out_expanded)

        self.raw_grad = self.raw_grad.to("cpu") if self.raw_grad != None else None

        self.calibrated = True
        out=self._bias_correction_quant_forward(x)
        del self.raw_input, self.raw_out, self.raw_grad
        return out 

class FPPTQSLBatchingQuantLinear_MinMax(FPPTQSLQuantLinear):
    def __init__(self, 
        in_features: int,
        out_features: int,
        bias: bool = True,
        mode = "raw",
        w_bit = 8,
        a_bit = 8,
        w_exponent_bit = 4, a_exponent_bit = 4,
        bias_bit = None,
        bias_correction = False,
        metric="L2_norm", search_round=1, eq_alpha=0, eq_beta=1, eq_n=100, parallel_eq_n=10, n_H=1, n_V=1, n_a=1):
        super().__init__(in_features, out_features, bias=bias, mode=mode, w_bit=w_bit, a_bit=a_bit, w_exponent_bit= w_exponent_bit, a_exponent_bit=a_exponent_bit, bias_bit=bias_bit, bias_correction=bias_correction, metric=metric, search_round=search_round, eq_alpha=eq_alpha, eq_beta=eq_beta, eq_n=eq_n, parallel_eq_n=parallel_eq_n, n_H=n_H, n_V=n_V, n_a=n_a)
        self.calib_size = None
        self.calib_batch_size = None
        self.calib_need_batching = False

    def _initialize_calib_parameters(self):
        """ 
        set parameters for feeding calibration data
        """
        self.calib_size = int(self.raw_input.shape[0])
        self.calib_batch_size = int(self.raw_input.shape[0])
        i = 0
        while True:
            numel = (2*(self.raw_input.numel()+self.raw_out.numel())/self.calib_size*self.calib_batch_size) # number of parameters on GPU
            self.parallel_eq_n = int((3*1024*1024*1024/4)//numel)
            if self.parallel_eq_n <= 1:
                self.calib_need_batching = True
                self.calib_batch_size //= 2
            else:
                break
    
    def _initialize_intervals(self):
        # weight intervals
        # print("adopt channel wise quantization for linear weight")
        ## channel wise
        self.n_V = self.out_features
        self.crb_rows = self.out_features // self.n_V
        w_maxval = self.weight.view(self.n_V, self.crb_rows,self.n_H,self.crb_cols).abs().amax([1,3],keepdim=True)
        self.w_interval=(2**self.w_exponent_bit - torch.log2(w_maxval) + math.log2(2 - 2 ** (-self.w_mantissa_bit)) - 1)
    
        # activation intervals
        tmp_a_intervals = []
        for b_st in range(0,self.calib_size,self.calib_batch_size):
            b_ed = min(self.calib_size, b_st+self.calib_batch_size)
            # print(f"self.weight.device {self.weight.device}")
            x_ = self.raw_input[b_st:b_ed].to(self.weight.device)
            print("tensor wise")
            x_maxval = x_.abs().max()
            a_interval_=(2**self.a_exponent_bit - torch.log2(x_maxval) + math.log2(2 - 2 ** (-self.a_mantissa_bit)) - 1).detach().view(1,1).repeat(self.n_a,1)
            tmp_a_intervals.append(a_interval_)
        self.a_interval = torch.cat(tmp_a_intervals, dim=1).amin(dim=1, keepdim=True)
        print(f"self.a_interval {self.a_interval.shape}")

    def _initialize_intervals_eval(self):
        self._initialize_calib_parameters()
        self.n_V = self.out_features
        self.crb_rows = self.out_features // self.n_V
        w_maxval = self.weight.view(self.n_V, self.crb_rows,self.n_H,self.crb_cols).abs().amax([1,3],keepdim=True)
        self.w_interval=(2**self.w_exponent_bit - torch.log2(w_maxval) + math.log2(2 - 2 ** (-self.w_mantissa_bit)) - 1)
    
        # activation intervals
        tmp_a_intervals = []
        for b_st in range(0,self.calib_size,self.calib_batch_size):
            b_ed = min(self.calib_size, b_st+self.calib_batch_size)
            # print(f"self.weight.device {self.weight.device}")
            x_ = self.raw_input[b_st:b_ed].to(self.weight.device)
            print("tensor wise")
            x_maxval = x_.abs().max()
            a_interval_=(2**self.a_exponent_bit - torch.log2(x_maxval) + math.log2(2 - 2 ** (-self.a_mantissa_bit)) - 1).detach().view(1,1).repeat(self.n_a,1)
            tmp_a_intervals.append(a_interval_)
        self.a_interval = torch.cat(tmp_a_intervals, dim=1).amin(dim=1, keepdim=True)
        self.calibrated = True

    def calibration_step2(self):
        """
        Only use cached raw inputs/outs/grads
        """
        self._initialize_calib_parameters()
        self._initialize_intervals()
        self.calibrated = True
        # self._bias_correction_quant_forward(self.raw_input.to(self.weight.device)) # debugging
        del self.raw_input, self.raw_out, self.raw_grad
        return None

class FPPTQSLBatchingQuantLinear_fpq_baseline(FPPTQSLQuantLinear):
    def __init__(self, 
        in_features: int,
        out_features: int,
        bias: bool = True,
        mode = "raw",
        w_bit = 8,
        a_bit = 8,
        w_exponent_bit = 4, a_exponent_bit = 4,
        bias_bit = None,
        bias_correction = False,
        metric="L2_norm", search_round=1, eq_alpha=0, eq_beta=1, eq_n=100, parallel_eq_n=10, n_H=1, n_V=1, n_a=1):
        super().__init__(in_features, out_features, bias=bias, mode=mode, w_bit=w_bit, a_bit=a_bit, w_exponent_bit= w_exponent_bit, a_exponent_bit=a_exponent_bit, bias_bit=bias_bit, bias_correction=bias_correction, metric=metric, search_round=search_round, eq_alpha=eq_alpha, eq_beta=eq_beta, eq_n=eq_n, parallel_eq_n=parallel_eq_n, n_H=n_H, n_V=n_V, n_a=n_a)
        self.calib_size = None
        self.calib_batch_size = None
        self.calib_need_batching = False
        self.w_maxval = None
        self.w_intervals = None
        self.a_maxval = None
        self.a_intervals = None

    def _initialize_calib_parameters(self):
        """ 
        set parameters for feeding calibration data
        """
        self.calib_size = int(self.raw_input.shape[0])
        self.calib_batch_size = int(self.raw_input.shape[0])
        i = 0
        while True:
            numel = (2*(self.raw_input.numel()+self.raw_out.numel())/self.calib_size*self.calib_batch_size) # number of parameters on GPU
            self.parallel_eq_n = int((3*1024*1024*1024/4)//numel)
            if self.parallel_eq_n <= 1:
                self.calib_need_batching = True
                self.calib_batch_size //= 2
            else:
                break
            
    def _initialize_intervals(self):
        # weight intervals 
        print("channel-wise weight")
        self.n_V = self.out_features
        self.crb_rows = self.out_features // self.n_V
        w_maxval = self.weight.view(self.n_V, self.crb_rows,self.n_H,self.crb_cols).abs().amax([1,3],keepdim=True)
        self.w_maxval = w_maxval
        self.w_interval=(2**self.w_exponent_bit - torch.log2(w_maxval) + math.log2(2 - 2 ** (-self.w_mantissa_bit)) - 1)
        self.w_intervals = []
        if self.w_bit == 8:
            for i in range(self.w_bit-3):
                M = i + 2
                E = self.w_bit - 1 - M
                self.w_intervals.append(2**E - torch.log2(self.w_maxval) + math.log2(2 - 2 ** (-M)) - 1)

        else:
            for i in range(self.w_bit-1):
                M = i
                E = self.w_bit - 1 - M
                self.w_intervals.append(2**E - torch.log2(self.w_maxval) + math.log2(2 - 2 ** (-M)) - 1)

        # activation intervals
        tmp_a_maxvals = []
        for b_st in range(0,self.calib_size,self.calib_batch_size):
            b_ed = min(self.calib_size, b_st+self.calib_batch_size)
            x_ = self.raw_input[b_st:b_ed].to(self.weight.device)
            x_maxval = x_.abs().max()
            tmp_a_maxvals.append(x_maxval)
        
        # print(f'tmp_a_intervals[0] {tmp_a_intervals[0].shape}')
        tmp_a_maxvals = torch.tensor(tmp_a_maxvals).to(x_.device)
        # print(f'tmp_a_maxvals {tmp_a_maxvals.shape}')
        self.a_maxval = tmp_a_maxvals.amax(dim=0, keepdim=True)
        self.a_interval = (2**self.a_exponent_bit - torch.log2(self.a_maxval) + math.log2(2 - 2 ** (-self.a_mantissa_bit)) - 1).detach().view(1,1).repeat(self.n_a,1)

        self.a_intervals = []
        if self.a_bit == 8:
            for i in range(self.a_bit-3):
                M = i + 2
                E = self.a_bit - 1 - M
                a_interval_=(2**E - torch.log2(self.a_maxval) + math.log2(2 - 2 ** (-M)) - 1).detach().view(1,1).repeat(self.n_a,1)

                self.a_intervals.append(a_interval_.clone())
        else:
            for i in range(self.a_bit-1):
                M = i
                E = self.a_bit - 1 - M
                a_interval_=(2**E - torch.log2(self.a_maxval) + math.log2(2 - 2 ** (-M)) - 1).detach().view(1,1).repeat(self.n_a,1)
                self.a_intervals.append(a_interval_.clone())

    def _initialize_intervals_eval(self):
        self._initialize_calib_parameters()
        print("channel-wise weight")
        self.n_V = self.out_features
        self.crb_rows = self.out_features // self.n_V
        w_maxval = self.weight.view(self.n_V, self.crb_rows,self.n_H,self.crb_cols).abs().amax([1,3],keepdim=True)
        self.w_maxval = w_maxval
        self.w_interval=(2**self.w_exponent_bit - torch.log2(w_maxval) + math.log2(2 - 2 ** (-self.w_mantissa_bit)) - 1)

        # activation intervals
        tmp_a_maxvals = []
        for b_st in range(0,self.calib_size,self.calib_batch_size):
            b_ed = min(self.calib_size, b_st+self.calib_batch_size)
            x_ = self.raw_input[b_st:b_ed].to(self.weight.device)
            x_maxval = x_.abs().max()
            tmp_a_maxvals.append(x_maxval)
        
        tmp_a_maxvals = torch.tensor(tmp_a_maxvals).to(x_.device)
        self.a_maxval = tmp_a_maxvals.amax(dim=0, keepdim=True)
        self.a_interval = (2**self.a_exponent_bit - torch.log2(self.a_maxval) + math.log2(2 - 2 ** (-self.a_mantissa_bit)) - 1).detach().view(1,1).repeat(self.n_a,1)

        self.calibrated = True

    def _get_similarity(self, tensor_raw, tensor_sim, metric=None, raw_grad=None):
        """
        tensor_raw: *, features
        tensor_sim: *, features
        similarity: *
        It's your job to calculate mean on * dims!
        """
        if metric == "cosine":
            similarity = F.cosine_similarity(tensor_raw, tensor_sim, dim=-1)
        else:
            if metric == "L1_norm":
                similarity = -torch.abs(tensor_raw - tensor_sim)
            elif metric == "linear_weighted_L2_norm":
                similarity = -tensor_raw.abs() * (tensor_raw - tensor_sim) ** 2
            elif metric == "square_weighted_L2_norm":
                similarity = -(tensor_raw * (tensor_raw - tensor_sim)) ** 2
            elif metric == "L2_norm":
                similarity = -(tensor_raw - tensor_sim) ** 2
            else:
                raise NotImplementedError(f"metric {metric} not implemented!")
            similarity = torch.mean(similarity, dim=-1)
        return similarity

    def _get_pearson_w(self, tensor_raw, tensor_sim):
        """
        Quick implementation of similarity-aware linear quantization
        tensor_sim: b,*,parallel_eq_n,n_V,crb_rows
        tensor_raw: b,*,1,n_V,crb_rows
        """
        b, parallel_eq_n, n_V = tensor_sim.shape[0],tensor_sim.shape[-3],tensor_sim.shape[-2]
        tensor_sim = tensor_sim.transpose(-1,-3).contiguous_().view(b,-1,n_V,parallel_eq_n)
        tensor_raw = tensor_raw.transpose(-1,-3).view(b,-1,n_V,1)
        tensor_sim_mean = tensor_sim.mean(dim=[0,1],keepdim=True)
        tensor_raw_mean = tensor_raw.mean(dim=[0,1],keepdim=True)
        similarity = torch.cosine_similarity(tensor_raw-tensor_raw_mean, tensor_sim-tensor_sim_mean, dim=1) # shape: b,n_V,parallel_eq_n
        similarity = similarity.permute(0,2,1).contiguous_()
        return similarity
    
    def _get_pearson_a(self, tensor_raw, tensor_sim):
        """
        Quick implementation of similarity-aware linear quantization
        tensor_sim: b,*,parallel_eq_n,oc
        tensor_raw: b,*,1,oc
        """
        b, parallel_eq_n = tensor_sim.shape[0],tensor_sim.shape[-2]
        tensor_sim = tensor_sim.transpose(-1,-2).contiguous_().view(b,-1,parallel_eq_n)
        tensor_raw = tensor_raw.transpose(-1,-2).view(b,-1,1)
        tensor_sim_mean = tensor_sim.mean(dim=[0,1],keepdim=True)
        tensor_raw_mean = tensor_raw.mean(dim=[0,1],keepdim=True)
        similarity = torch.cosine_similarity(tensor_raw-tensor_raw_mean, tensor_sim-tensor_sim_mean, dim=1) # shape: b,parallel_eq_n
        return similarity

    def _search_best_w_interval(self, weight_interval_candidates):
        
        # tmp_w_interval = self.w_interval.unsqueeze(0) # shape: 1,n_V,1,n_H,1
        # print(f"weight_interval_candidates shape {weight_interval_candidates.shape}")
        for man in range(weight_interval_candidates.shape[0]):
            tmp_w_interval = self.w_intervals[man].unsqueeze(0) # shape: 1,n_V,1,n_H,1
            for h in range(self.n_H):
                batch_similarities = [] # similarities, need to concatenate and calculate sum (equivalent to mean with argmax)
                # print(f"before search E{self.w_bit-1-man}M{man} self.w_intervals[man] {self.w_intervals[man][0][0]}")
                for b_st in range(0, self.calib_size, self.calib_batch_size):
                    b_ed = min(self.calib_size, b_st + self.calib_batch_size)
                    x = self.raw_input[b_st:b_ed].to(self.weight.device)
                    raw_out_expanded = self.raw_out[b_st:b_ed].to(self.weight.device).unsqueeze(-2) # shape: b,*,1,oc
                    raw_out_expanded = torch.cat(torch.chunk(raw_out_expanded.unsqueeze(-2), chunks=self.n_V, dim=-1), dim=-2) # shape: b,*,1,n_V,crb_rows
                    raw_grad = self.raw_grad
                    similarities = []
                    for p_st in range(0,self.eq_n,self.parallel_eq_n):
                        p_ed = min(self.eq_n, p_st+self.parallel_eq_n)
                        cur_w_interval = tmp_w_interval.repeat(p_ed-p_st,1,1,1,1)
                        # print(f"cur_w_interval {cur_w_interval.shape}")
                        cur_w_interval[:,:,:,h:h+1,:] = weight_interval_candidates[man][p_st:p_ed,:,:,h:h+1,:]
                        # quantize weight and bias 
                        w_sim = self.weight.view(self.n_V,self.crb_rows,self.n_H,self.crb_cols).unsqueeze(0) # shape: 1,n_V,crb_rows,n_H,crb_cols

                        if self.w_bit == 8:
                            w, cur_w_scale = self.get_scale(w_sim, bits = self.w_bit, mantissa_bit= man+2, bias= cur_w_interval)
                        else:
                            w, cur_w_scale = self.get_scale(w_sim, bits = self.w_bit, mantissa_bit= man, bias= cur_w_interval)
    
    
                        w_sim = (w/cur_w_scale).round_().mul_(cur_w_scale) # shape: parallel_eq_n,n_V,crb_rows,n_H,crb_cols
                        w_sim = w_sim.view(-1,self.in_features) # shape: parallel_eq_n*oc,ic
                        bias_sim = self.bias.repeat(p_ed-p_st) if self.bias is not None else None
                        # quantize input
                        x_sim = self.quant_input(x)
                        # calculate similarity and store them
                        out_sim = F.linear(x_sim, w_sim, bias_sim) # shape: b,*,parallel_eq_n*oc
                        out_sim = torch.cat(torch.chunk(out_sim.unsqueeze(-2), chunks=p_ed-p_st, dim=-1), dim=-2) # shape: b,*,parallel_eq_n,oc
                        out_sim = torch.cat(torch.chunk(out_sim.unsqueeze(-2), chunks=self.n_V, dim=-1), dim=-2) # shape: b,*,parallel_eq_n,n_V,crb_rows
                        if self.metric != "pearson":
                            similarity = self._get_similarity(raw_out_expanded, out_sim, self.metric, raw_grad) # shape: b,*,parallel_eq_n,n_V
                            if len(similarity.shape) > 3:
                                similarity = torch.mean(similarity, dim=list(range(1,len(similarity.shape)-2))) # shape: b, parallel_eq_n, n_V
                        else:
                            similarity = self._get_pearson_w(raw_out_expanded, out_sim)
                        similarity = similarity.sum(dim=0, keepdim=True) # shape: 1, parallel_eq_n, n_V
                        similarities.append(similarity)
                    # store best weight interval of h into tmp_w_interval
                    similarities = torch.cat(similarities, dim=1) # shape: 1, eq_n, n_V
                    batch_similarities.append(similarities)
                batch_similarities = torch.cat(batch_similarities, dim=0).sum(dim=0, keepdim=False) # shape: eq_n, n_V
                h_best_index = batch_similarities.argmax(dim=0).reshape(1,-1,1,1,1) # shape: 1,n_V,1,1,1
                tmp_w_interval[:,:,:,h:h+1,:] = torch.gather(weight_interval_candidates[man][:,:,:,h:h+1,:],dim=0,index=h_best_index)
            self.w_intervals[man] = tmp_w_interval.squeeze(dim=0)

    def _search_best_w_format(self):
        
        # print(f"before search linear weight E{self.w_exponent_bit}M{self.w_mantissa_bit}")
        
        # format candidate
        w_mantissa_bits_candidate = [i for i in range(self.w_bit-1)]
        
        batch_similarities = [] # similarities, need to concatenate and calculate sum (equivalent to mean with argmax)
        for b_st in range(0, self.calib_size, self.calib_batch_size):
            b_ed = min(self.calib_size, b_st + self.calib_batch_size)
            x = self.raw_input[b_st:b_ed].to(self.weight.device)
            raw_out = self.raw_out[b_st:b_ed].to(self.weight.device) # shape: b,*,1,oc
            raw_grad = self.raw_grad
            similarities = []
            # quantize input
            x_sim = self.quant_input(x)
            for w_mantissa_bit in w_mantissa_bits_candidate:
                if self.w_bit == 8:
                    w_mantissa_bit = w_mantissa_bit + 2
                else:
                    w_mantissa_bit = w_mantissa_bit
                w_sim = self.weight.view(self.n_V,self.crb_rows,self.n_H,self.crb_cols)
                w,cur_w_scale = self.get_scale(w_sim, bits = self.w_bit, mantissa_bit= w_mantissa_bit, bias= self.w_intervals[w_mantissa_bit])
                w_sim = (w/cur_w_scale).round_().mul_(cur_w_scale)
                w_sim = w_sim.view(-1,self.in_features)
                bias_sim = self.bias if self.bias is not None else None
                out_sim = F.linear(x_sim, w_sim, bias_sim) # shape: B,*,oc
                similarity = self._get_similarity(raw_out, out_sim, self.metric, raw_grad) #B,*,oc
                # print(f"weight similarity shape {similarity.shape}")
                similarity = torch.mean(similarity) # shape: 1
                similarities.append(similarity)
            similarities = torch.tensor(similarities)
            # print(f"weight similarities {similarities}")
            batch_similarities.append(similarities)
        batch_similarities = torch.vstack(batch_similarities)
        # print(f"weight batch_similarities {batch_similarities}")
        best_mantissa_bit = batch_similarities.sum(dim=0, keepdim=True).argmax(dim=1).item()
        
        if self.w_bit == 8:
            self.w_mantissa_bit = torch.tensor(best_mantissa_bit + 2).to(self.weight.device)
            self.w_exponent_bit = torch.tensor(self.w_bit - 1 - best_mantissa_bit).to(self.weight.device)    
        
        else:
            self.w_mantissa_bit = torch.tensor(best_mantissa_bit).to(self.weight.device)
            self.w_exponent_bit = torch.tensor(self.w_bit - 1 - self.w_mantissa_bit).to(self.weight.device)  
              
        self.w_interval = self.w_intervals[self.w_mantissa_bit]
        # print(f"search result E{self.w_exponent_bit}M{self.w_mantissa_bit}")
        # print(f"after calibrate bias {self.w_interval[[0,10,30,40,50]]}")
        # print("finish searching fp format for linear weight")

    def _search_best_a_interval(self, input_interval_candidates):
        
        # print(f"input_interval_candidates shape {input_interval_candidates.shape}")
        for man in range(input_interval_candidates.shape[0]):
            tmp_a_interval = self.a_intervals[man].unsqueeze(-1) # shape: n_a,1,1
            # print(f"tmp_a_interval.shape {tmp_a_interval.shape}")
            for a in range(self.n_a):
                batch_similarities = [] # similarities, need to concatenate and calculate sum (equivalent to mean with argmax)
                for b_st in range(0, self.calib_size, self.calib_batch_size):
                    b_ed = min(self.calib_size, b_st + self.calib_batch_size)
                    x = self.raw_input[b_st:b_ed].to(self.weight.device)
                    raw_out_expanded = self.raw_out[b_st:b_ed].to(self.weight.device).unsqueeze(-2) # shape: b,*,1,oc
                    raw_grad = self.raw_grad
                    similarities = []
                    for p_st in range(0,self.eq_n,self.parallel_eq_n):
                        p_ed = min(self.eq_n, p_st+self.parallel_eq_n)
                        cur_a_interval = tmp_a_interval.repeat(1,1,p_ed-p_st) # shape: n_a,1,parallel_eq_n
                        cur_a_interval[a:a+1,:,:] = input_interval_candidates[man][a:a+1,:,p_st:p_ed]
                        # quantize weight and bias 
                        w_sim, bias_sim = self.quant_weight_bias()
                        # quantize input
                        x_sim=torch.cat(torch.chunk(x.unsqueeze(-2), chunks=self.n_a, dim=-1), dim=-2).unsqueeze(-1)
                        
                        if self.a_bit == 8:
                            # print(f"CUR a E{self.a_bit - 1 - man -2}M{man+2}")
                            cur_a, cur_a_scale = self.get_scale(x_sim, bits = self.a_bit, mantissa_bit= man+2, bias= cur_a_interval)
                        else:
                            cur_a, cur_a_scale = self.get_scale(x_sim, bits = self.a_bit, mantissa_bit= man, bias= cur_a_interval)

                        x_sim=(cur_a/(cur_a_scale)).round_()*(cur_a_scale) # shape: b,*,n_a,crb_acts,parallel_eq_n
                        x_sim = x_sim.permute(*list(range(len(x_sim.shape)-3)),-1,-3,-2).reshape(*x.shape[:-1],p_ed-p_st,x.shape[-1]) # shape: b,*,parallel_eq_n,ic
                        # calculate similarity and store them
                        out_sim = F.linear(x_sim, w_sim, bias_sim) # shape: b,*,parallel_eq_n,oc
                        if self.metric != "pearson":
                            similarity = self._get_similarity(raw_out_expanded, out_sim, self.metric, raw_grad) # shape: b,*,parallel_eq_n
                            if len(similarity.shape) > 2:
                                similarity = torch.mean(similarity, dim=list(range(1,len(similarity.shape)-1))) # shape: b, parallel_eq_n
                        else:
                            similarity = self._get_pearson_a(raw_out_expanded, out_sim)
                        similarity = torch.sum(similarity, dim=0, keepdim=True) # shape: 1, parallel_eq_n
                        similarities.append(similarity)
                    # store best input interval and store in tmp_a_interval
                    similarities = torch.cat(similarities, dim=1) # shape: 1, eq_n
                    batch_similarities.append(similarities)
                batch_similarities = torch.cat(batch_similarities, dim=0).sum(dim=0, keepdim=False) # shape: eq_n
                a_best_index = batch_similarities.argmax(dim=0, keepdim=True).reshape(1,1,-1)
                tmp_a_interval[a:a+1,:,:] = torch.gather(input_interval_candidates[man][a:a+1,:,:],dim=2,index=a_best_index)
            self.a_intervals[man] = tmp_a_interval.squeeze(-1)

    def _search_best_a_format(self):
        
        batch_similarities = [] # similarities, need to concatenate and calculate sum (equivalent to mean with argmax)

        # format candidate
        if self.a_bit == 8:
            a_mantissa_bits_candidate = [i for i in range(self.a_bit-3)]
        else:
            a_mantissa_bits_candidate = [i for i in range(self.a_bit-1)]
        # quantize input
        w_sim, bias_sim = self.quant_weight_bias()
        # print(f"before search linear activation E{self.a_exponent_bit}M{self.a_mantissa_bit}")
        for b_st in range(0, self.calib_size, self.calib_batch_size):
            b_ed = min(self.calib_size, b_st + self.calib_batch_size)
            x = self.raw_input[b_st:b_ed].to(self.weight.device)
            raw_out = self.raw_out[b_st:b_ed].to(self.weight.device) # shape: b,*,oc
            raw_grad = self.raw_grad
            similarities = []
            
            for a_mantissa_bit in a_mantissa_bits_candidate:
                if self.a_bit == 8:
                    a_mantissa_bit = a_mantissa_bit + 2
                    
                x_sim = torch.cat(torch.chunk(x.unsqueeze(-2), chunks=self.n_a, dim=-1), dim=-2)
                cur_a, cur_a_scale = self.get_scale(x_sim, bits = self.a_bit, mantissa_bit= a_mantissa_bit, bias= self.a_intervals[a_mantissa_bit])
                x_sim=(cur_a/(cur_a_scale)).round_()*(cur_a_scale) # shape: B,*,n_a,crb_acts
                # print(f"x_sim shape {x_sim.shape}")
                if len(x.shape) == 3:
                    x_sim = x_sim.view(x.shape[0],x.shape[1],x.shape[2])
                else:
                    # print(f"x {x.shape}")
                    # print(f"raw_out {raw_out.shape}")
                    x_sim = x_sim.view(x.shape[0],1,x.shape[1])
                out_sim = F.linear(x_sim, w_sim, bias_sim) # shape: B,*,oc 
                # print(f"E{self.a_bit - 1 - a_mantissa_bit}M{a_mantissa_bit}")
                # print(f"search act out_sim {out_sim.shape}")
                # print(f"search act out_sim {out_sim[0][2][0:10]}")
                # print(f"raw_out {raw_out[0][2][0:10]}")
                similarity = self._get_similarity(raw_out, out_sim, self.metric, raw_grad) #B,*,oc
                # print(f"activation similarity shape {similarity.shape}")
                similarity = torch.mean(similarity)
                # print(f"similarity: {similarity}")
                similarities.append(similarity)
            similarities = torch.tensor(similarities)
            batch_similarities.append(similarities)
                
        batch_similarities = torch.vstack(batch_similarities)
        best_mantissa_bit = batch_similarities.sum(dim=0, keepdim=True).argmax(dim=1).item()
        
        if self.a_bit == 8:
            self.a_mantissa_bit = torch.tensor(best_mantissa_bit + 2).to(self.weight.device)
            self.a_exponent_bit = torch.tensor(self.a_bit - 1 - best_mantissa_bit).to(self.weight.device)    
        
        else:
            self.a_mantissa_bit = torch.tensor(best_mantissa_bit).to(self.weight.device)
            self.a_exponent_bit = torch.tensor(self.a_bit - 1 - best_mantissa_bit).to(self.weight.device)    
         
        self.a_interval = self.a_intervals[self.a_mantissa_bit]
        # print(f"search result linear activation E{self.a_exponent_bit}M{self.a_mantissa_bit}")
        # print(f"after calibrate bias {self.w_interval[[0,10,30,40,50]]}")
        # print("finish searching fp format for linear activation")

    def calibration_step2(self):
        """
        Only use cached raw inputs/outs/grads
        """
        self._initialize_calib_parameters()
        self._initialize_intervals()

        # prepare weight intervals and similarities
        weight_interval_candidates = []
        if self.w_bit == 8:
            for m in range(self.w_bit-3):
                weight_interval_candidate = torch.tensor([self.eq_alpha + i*(self.eq_beta - self.eq_alpha)/self.eq_n for i in range(self.eq_n + 1)]).to(self.weight.device).view(-1,1,1,1,1) * self.w_intervals[m].unsqueeze(0)
                weight_interval_candidates.append(weight_interval_candidate.unsqueeze(0)) # shape: num_man_options,eq_n,n_V,1,n_H,1
        else:
            for m in range(self.w_bit-1):
                weight_interval_candidate = torch.tensor([self.eq_alpha + i*(self.eq_beta - self.eq_alpha)/self.eq_n for i in range(self.eq_n + 1)]).to(self.weight.device).view(-1,1,1,1,1) * self.w_intervals[m].unsqueeze(0)
                weight_interval_candidates.append(weight_interval_candidate.unsqueeze(0)) # shape: num_man_options,eq_n,n_V,1,n_H,1
        weight_interval_candidates = torch.vstack(weight_interval_candidates)

        input_interval_candidates = []
        if self.a_bit == 8:
            for m in range(self.a_bit-3): 
                input_interval_candidate = torch.tensor([self.eq_alpha + i*(self.eq_beta - self.eq_alpha)/self.eq_n for i in range(self.eq_n + 1)]).to(self.weight.device).view(1,1,-1) * self.a_intervals[m].unsqueeze(-1)
                input_interval_candidates.append(input_interval_candidate.unsqueeze(0)) # shape: n_a,1,eq_n
        else:
            for m in range(self.a_bit-1): 
                input_interval_candidate = torch.tensor([self.eq_alpha + i*(self.eq_beta - self.eq_alpha)/self.eq_n for i in range(self.eq_n + 1)]).to(self.weight.device).view(1,1,-1) * self.a_intervals[m].unsqueeze(-1)
                input_interval_candidates.append(input_interval_candidate.unsqueeze(0)) # shape: n_a,1,eq_n
        input_interval_candidates = torch.vstack(input_interval_candidates)
        
        for e in range(self.search_round):
            # search for best weight interval
            self._search_best_w_interval(weight_interval_candidates)
            # search for best weight format
            self._search_best_w_format()
            # search for best input interval
            self._search_best_a_interval(input_interval_candidates)
            # search for best input format
            self._search_best_a_format()


        self.calibrated = True
        del self.raw_input, self.raw_out, self.raw_grad
        return None

class FPPTQSLBatchingQuantLinear_fpq(FPPTQSLQuantLinear):
    def __init__(self, 
        in_features: int,
        out_features: int,
        bias: bool = True,
        mode = "raw",
        w_bit = 8,
        a_bit = 8,
        w_exponent_bit = 4, a_exponent_bit = 4,
        bias_bit = None,
        bias_correction = False,
        metric="L2_norm", search_round=1, eq_alpha=0, eq_beta=1, eq_n=100, parallel_eq_n=10, n_H=1, n_V=1, n_a=1):
        super().__init__(in_features, out_features, bias=bias, mode=mode, w_bit=w_bit, a_bit=a_bit, w_exponent_bit= w_exponent_bit, a_exponent_bit=a_exponent_bit, bias_bit=bias_bit, bias_correction=bias_correction, metric=metric, search_round=search_round, eq_alpha=eq_alpha, eq_beta=eq_beta, eq_n=eq_n, parallel_eq_n=parallel_eq_n, n_H=n_H, n_V=n_V, n_a=n_a)
        self.calib_size = None
        self.calib_batch_size = None
        self.calib_need_batching = False
        self.w_maxval = None
        self.w_intervals = None
        self.a_maxval = None
        self.register_buffer('a_bias',None)
        self.a_biases = None ## fix channel-wise biases
        self.a_intervals = None ## now search for tensor scale not the channel-wise biases
        self.register_buffer('a_interval_zero_point',None)
        self.a_intervals_zero_point = None
        self.n_ls = 1

    def _initialize_calib_parameters(self):
        """ 
        set parameters for feeding calibration data
        """
        self.calib_size = int(self.raw_input.shape[0])
        self.calib_batch_size = int(self.raw_input.shape[0])
        i = 0
        while True:
            numel = (2*(self.raw_input.numel()+self.raw_out.numel())/self.calib_size*self.calib_batch_size) # number of parameters on GPU
            self.parallel_eq_n = int((3*1024*1024*1024/4)//numel)
            if self.parallel_eq_n <= 1:
                self.calib_need_batching = True
                self.calib_batch_size //= 2
            else:
                break
    
    def _initialize_intervals(self):
        # weight intervals 
        ## channel wise
        ## specific for QKV
        if self.n_V != 1:
            # print("tackling QKV linear")
            self.n_ls = 3 # number of tensor scale    
        print("channel-wise weight")
        self.n_V = self.out_features
        self.crb_rows = self.out_features // self.n_V
        w_maxval = self.weight.view(self.n_V, self.crb_rows,self.n_H,self.crb_cols).abs().amax([1,3],keepdim=True)
        self.w_maxval = w_maxval
        self.w_interval=(2**self.w_exponent_bit - torch.log2(w_maxval) + math.log2(2 - 2 ** (-self.w_mantissa_bit)) - 1)
        self.w_intervals = []
        if self.w_bit == 8:
            for i in range(self.w_bit-3):
                M = i + 2
                E = self.w_bit - 1 - M
                self.w_intervals.append(2**E - torch.log2(self.w_maxval) + math.log2(2 - 2 ** (-M)) - 1)

        
        else:
            for i in range(self.w_bit-1):
                M = i
                E = self.w_bit - 1 - M
                self.w_intervals.append(2**E - torch.log2(self.w_maxval) + math.log2(2 - 2 ** (-M)) - 1)
        
        # activation intervals
        tmp_a_maxvals = []
        for b_st in range(0,self.calib_size,self.calib_batch_size):
            b_ed = min(self.calib_size, b_st+self.calib_batch_size)
            x_ = self.raw_input[b_st:b_ed].to(self.weight.device)
            self.n_a = self.in_features
            self.crb_acts = self.in_features // self.n_a
            x_maxval = x_.view(*x_.shape[:-1],self.n_a,self.crb_acts).abs().amax(list(range(len(x_.shape)-1))+[-1],keepdim=False).unsqueeze(-1)
            tmp_a_maxvals.append(x_maxval)

        
        tmp_a_maxvals = torch.cat(tmp_a_maxvals, dim=1)
        self.a_maxval = tmp_a_maxvals.amax(dim=1, keepdim=True)
        self.a_bias = 2**self.a_exponent_bit - torch.log2(self.a_maxval) + math.log2(2 - 2 ** (-self.a_mantissa_bit)) - 1


        self.a_interval = (self.a_bias.min())
        self.a_interval_zero_point = torch.round(self.a_interval)

        self.a_biases = []
        self.a_intervals = []
        self.a_intervals_zero_point = []
        if self.a_bit == 8:
            for i in range(self.a_bit-3):
                M = i + 2
                E = self.a_bit - 1 - M
                cur_a_bias = (2**E - torch.log2(self.a_maxval) + math.log2(2 - 2 ** (-M)) - 1)
                self.a_biases.append(cur_a_bias)
                cur_a_interval = (cur_a_bias.min())
                self.a_intervals.append(cur_a_interval.reshape(1,1))
                self.a_intervals_zero_point.append(torch.round(cur_a_bias.min()))
            
        else:
            for i in range(self.a_bit-1):
                M = i
                E = self.a_bit - 1 - M
                cur_a_bias = (2**E - torch.log2(self.a_maxval) + math.log2(2 - 2 ** (-M)) - 1)
                self.a_biases.append(cur_a_bias)
                cur_a_interval = (cur_a_bias.min())
                self.a_intervals.append(cur_a_interval.reshape(1,1))
                self.a_intervals_zero_point.append(torch.round(cur_a_bias.min()))

    def _initialize_intervals_eval(self):
        self._initialize_calib_parameters()
        # weight intervals 
        ## channel wise
        ## specific for QKV
        if self.n_V != 1:
            # print("tackling QKV linear")
            self.n_ls = 3 # number of tensor scale    
        print("channel-wise weight")
        self.n_V = self.out_features
        self.crb_rows = self.out_features // self.n_V
        w_maxval = self.weight.view(self.n_V, self.crb_rows,self.n_H,self.crb_cols).abs().amax([1,3],keepdim=True)
        self.w_maxval = w_maxval
        self.w_interval=(2**self.w_exponent_bit - torch.log2(w_maxval) + math.log2(2 - 2 ** (-self.w_mantissa_bit)) - 1)        
        # activation intervals
        tmp_a_maxvals = []
        for b_st in range(0,self.calib_size,self.calib_batch_size):
            b_ed = min(self.calib_size, b_st+self.calib_batch_size)
            x_ = self.raw_input[b_st:b_ed].to(self.weight.device)
            self.n_a = self.in_features
            self.crb_acts = self.in_features // self.n_a
            x_maxval = x_.view(*x_.shape[:-1],self.n_a,self.crb_acts).abs().amax(list(range(len(x_.shape)-1))+[-1],keepdim=False).unsqueeze(-1)
            tmp_a_maxvals.append(x_maxval)

        tmp_a_maxvals = torch.cat(tmp_a_maxvals, dim=1)
        self.a_maxval = tmp_a_maxvals.amax(dim=1, keepdim=True)
        self.a_bias = 2**self.a_exponent_bit - torch.log2(self.a_maxval) + math.log2(2 - 2 ** (-self.a_mantissa_bit)) - 1

        self.a_interval = (self.a_bias.min()).view(1,1)
        self.a_interval_zero_point = torch.round(self.a_interval).view(1,1)
        self.calibrated = True


    def get_maxval_from_bias(self, rescale_bias, act_or_weight):
        
        
        if act_or_weight == 0:
            
            return (2 - 2 ** (-self.a_mantissa_bit)) * 2 ** (
                2**self.a_exponent_bit - 1 - rescale_bias
            )
        elif act_or_weight == 1:
            
            return (2 - 2 ** (-self.w_mantissa_bit)) * 2 ** (
                2**self.w_exponent_bit - 1 - rescale_bias
            )

    def get_log_scale(self, x ,act_or_weight):
        
        if act_or_weight == 0:
            
            a_bias = self.a_bias
            a_bias = torch.clamp(torch.round(a_bias), torch.round(self.a_interval),  torch.round(self.a_interval) + 2**(self.a_exponent_bit) - 1 ) - self.a_interval_zero_point + self.a_interval
            a_bias = a_bias.float()
            a_maxval = self.get_maxval_from_bias(rescale_bias = a_bias, act_or_weight=0)
            a_minval = -a_maxval
            a = torch.min(torch.max(x, a_minval), a_maxval)
                        
            a_log_scales = torch.clamp((torch.floor(torch.log2(torch.abs(a)) + a_bias)).detach(), 1.0)
            return a, 2.0 ** (a_log_scales - self.a_mantissa_bit - a_bias)
        
        elif act_or_weight == 1:
            
            w_bias = self.w_interval
            w_bias = w_bias.float()
            w_maxval = self.get_maxval_from_bias(w_bias, 1)
            w_minval = -w_maxval
            w = torch.min(torch.max(x, w_minval), w_maxval)
            w_log_scales = torch.clamp((torch.floor(torch.log2(torch.abs(w)) + w_bias)).detach(), 1.0)
            return w, 2.0 ** (w_log_scales - self.w_mantissa_bit - w_bias)

    def get_w_scale(self, input, bits, mantissa_bit, bias):
        
        M = mantissa_bit
        E = bits - 1 - M
        bias = bias.float()
        maxval = (2 - 2 ** (-M)) * 2 ** (
                2**E - 1 - bias
            )

        minval = -maxval
        input = torch.min(torch.max(input, minval), maxval)
        input_log_scales = torch.clamp((torch.floor(torch.log2(torch.abs(input)) + bias)).detach(), 1.0)
        return input, 2.0 ** (input_log_scales - M - bias)
    
    def get_scale(self, input, bits, mantissa_bit, bias, tensor_scale, tensor_scale_zero_point):
        
        M = mantissa_bit
        E = bits - 1 - M
        
        rescale_bias = torch.clamp(torch.round(bias), torch.round(tensor_scale),  torch.round(tensor_scale) + 2**E - 1) - tensor_scale_zero_point + tensor_scale
        rescale_bias = rescale_bias.float()

        maxval = (2 - 2 ** (-M)) * 2 ** (
                2**E - 1 - rescale_bias
            )

        minval = -maxval
        input = torch.min(torch.max(input, minval), maxval)
        
        input_log_scales = torch.clamp((torch.floor(torch.log2(torch.abs(input)) + rescale_bias)).detach(), 1.0)

        return input, 2.0 ** (input_log_scales - M - rescale_bias)        

    def _get_similarity(self, tensor_raw, tensor_sim, metric=None, raw_grad=None):
        """
        tensor_raw: *, features
        tensor_sim: *, features
        similarity: *
        It's your job to calculate mean on * dims!
        """
        if metric == "cosine":
            similarity = F.cosine_similarity(tensor_raw, tensor_sim, dim=-1)
        else:
            if metric == "L1_norm":
                similarity = -torch.abs(tensor_raw - tensor_sim)
            elif metric == "L2_norm":
                similarity = -(tensor_raw - tensor_sim) ** 2
            elif metric == "linear_weighted_L2_norm":
                similarity = -tensor_raw.abs() * (tensor_raw - tensor_sim) ** 2
            elif metric == "square_weighted_L2_norm":
                similarity = -(tensor_raw * (tensor_raw - tensor_sim)) ** 2
            else:
                raise NotImplementedError(f"metric {metric} not implemented!")
            similarity = torch.mean(similarity, dim=-1)
        return similarity

    def _get_pearson_w(self, tensor_raw, tensor_sim):
        """
        Quick implementation of similarity-aware linear quantization
        tensor_sim: b,*,parallel_eq_n,n_V,crb_rows
        tensor_raw: b,*,1,n_V,crb_rows
        """
        b, parallel_eq_n, n_V = tensor_sim.shape[0],tensor_sim.shape[-3],tensor_sim.shape[-2]
        tensor_sim = tensor_sim.transpose(-1,-3).contiguous_().view(b,-1,n_V,parallel_eq_n)
        tensor_raw = tensor_raw.transpose(-1,-3).view(b,-1,n_V,1)
        tensor_sim_mean = tensor_sim.mean(dim=[0,1],keepdim=True)
        tensor_raw_mean = tensor_raw.mean(dim=[0,1],keepdim=True)
        similarity = torch.cosine_similarity(tensor_raw-tensor_raw_mean, tensor_sim-tensor_sim_mean, dim=1) # shape: b,n_V,parallel_eq_n
        similarity = similarity.permute(0,2,1).contiguous_()
        return similarity
    
    def _get_pearson_a(self, tensor_raw, tensor_sim):
        """
        Quick implementation of similarity-aware linear quantization
        tensor_sim: b,*,parallel_eq_n,oc
        tensor_raw: b,*,1,oc
        """
        b, parallel_eq_n = tensor_sim.shape[0],tensor_sim.shape[-2]
        tensor_sim = tensor_sim.transpose(-1,-2).contiguous_().view(b,-1,parallel_eq_n)
        tensor_raw = tensor_raw.transpose(-1,-2).view(b,-1,1)
        tensor_sim_mean = tensor_sim.mean(dim=[0,1],keepdim=True)
        tensor_raw_mean = tensor_raw.mean(dim=[0,1],keepdim=True)
        similarity = torch.cosine_similarity(tensor_raw-tensor_raw_mean, tensor_sim-tensor_sim_mean, dim=1) # shape: b,parallel_eq_n
        return similarity

    def _search_best_w_interval(self, weight_interval_candidates):
        
        # print(f"weight_interval_candidates shape {weight_interval_candidates.shape}")
        for man in range(weight_interval_candidates.shape[0]):
            # print(f"CUR w E{self.w_bit - 1 - man}M{man}")
            tmp_w_interval = self.w_intervals[man].unsqueeze(0) # shape: 1,n_V,1,n_H,1
            for h in range(self.n_H):
                batch_similarities = [] # similarities, need to concatenate and calculate sum (equivalent to mean with argmax)
                # print(f"before search E{self.w_bit-1-man}M{man} self.w_intervals[man] {self.w_intervals[man][0][0]}")
                for b_st in range(0, self.calib_size, self.calib_batch_size):
                    b_ed = min(self.calib_size, b_st + self.calib_batch_size)
                    x = self.raw_input[b_st:b_ed].to(self.weight.device)
                    raw_out_expanded = self.raw_out[b_st:b_ed].to(self.weight.device).unsqueeze(-2) # shape: b,*,1,oc
                    raw_out_expanded = torch.cat(torch.chunk(raw_out_expanded.unsqueeze(-2), chunks=self.n_V, dim=-1), dim=-2) # shape: b,*,1,n_V,crb_rows
                    raw_grad = self.raw_grad
                    similarities = []
                    
                    for p_st in range(0,self.eq_n,self.parallel_eq_n):
                        p_ed = min(self.eq_n, p_st+self.parallel_eq_n)
                        cur_w_interval = tmp_w_interval.repeat(p_ed-p_st,1,1,1,1)
                        # print(f"cur_w_interval {cur_w_interval.shape}")
                        cur_w_interval[:,:,:,h:h+1,:] = weight_interval_candidates[man][p_st:p_ed,:,:,h:h+1,:]
                        # quantize weight and bias 
                        w_sim = self.weight.view(self.n_V,self.crb_rows,self.n_H,self.crb_cols).unsqueeze(0) # shape: 1,n_V,crb_rows,n_H,crb_cols
                        
                        if self.w_bit == 8:
                            w, cur_w_scale = self.get_w_scale(w_sim, bits = self.w_bit, mantissa_bit= man+2, bias= cur_w_interval)
                        else:
                            w, cur_w_scale = self.get_w_scale(w_sim, bits = self.w_bit, mantissa_bit= man, bias= cur_w_interval)
    
                        w_sim = (w/cur_w_scale).round_().mul_(cur_w_scale) # shape: parallel_eq_n,n_V,crb_rows,n_H,crb_cols
                        w_sim = w_sim.view(-1,self.in_features) # shape: parallel_eq_n*oc,ic
                        bias_sim = self.bias.repeat(p_ed-p_st) if self.bias is not None else None
                        # quantize input
                        x_sim = self.quant_input(x)
                        # calculate similarity and store them
                        out_sim = F.linear(x_sim, w_sim, bias_sim) # shape: b,*,parallel_eq_n*oc
                        out_sim = torch.cat(torch.chunk(out_sim.unsqueeze(-2), chunks=p_ed-p_st, dim=-1), dim=-2) # shape: b,*,parallel_eq_n,oc
                        out_sim = torch.cat(torch.chunk(out_sim.unsqueeze(-2), chunks=self.n_V, dim=-1), dim=-2) # shape: b,*,parallel_eq_n,n_V,crb_rows
                        if self.metric != "pearson":
                            similarity = self._get_similarity(raw_out_expanded, out_sim, self.metric, raw_grad) # shape: b,*,parallel_eq_n,n_V
                            if len(similarity.shape) > 3:
                                similarity = torch.mean(similarity, dim=list(range(1,len(similarity.shape)-2))) # shape: b, parallel_eq_n, n_V
                        else:
                            similarity = self._get_pearson_w(raw_out_expanded, out_sim)
                        similarity = similarity.sum(dim=0, keepdim=True) # shape: 1, parallel_eq_n, n_V
                        similarities.append(similarity)
                    # store best weight interval of h into tmp_w_interval
                    similarities = torch.cat(similarities, dim=1) # shape: 1, eq_n, n_V
                    batch_similarities.append(similarities)
                batch_similarities = torch.cat(batch_similarities, dim=0).sum(dim=0, keepdim=False) # shape: eq_n, n_V
                h_best_index = batch_similarities.argmax(dim=0).reshape(1,-1,1,1,1) # shape: 1,n_V,1,1,1
                tmp_w_interval[:,:,:,h:h+1,:] = torch.gather(weight_interval_candidates[man][:,:,:,h:h+1,:],dim=0,index=h_best_index)
            self.w_intervals[man] = tmp_w_interval.squeeze(dim=0)

    def _search_best_w_format(self):
        
        # print(f"before search linear weight E{self.w_exponent_bit}M{self.w_mantissa_bit}")
        
        # format candidate
        if self.w_bit == 8:
            w_mantissa_bits_candidate = [i for i in range(self.w_bit-3)]
        else:
            w_mantissa_bits_candidate = [i for i in range(self.w_bit-1)]
        
        batch_similarities = [] # similarities, need to concatenate and calculate sum (equivalent to mean with argmax)
        for b_st in range(0, self.calib_size, self.calib_batch_size):
            b_ed = min(self.calib_size, b_st + self.calib_batch_size)
            x = self.raw_input[b_st:b_ed].to(self.weight.device)
            raw_out = self.raw_out[b_st:b_ed].to(self.weight.device) # shape: b,*,1,oc
            raw_grad = self.raw_grad
            similarities = []
            # quantize input
            x_sim = self.quant_input(x)
            
            for w_mantissa_bit in w_mantissa_bits_candidate:
                if self.w_bit == 8:
                    shift_w_mantissa_bit = w_mantissa_bit + 2
                else:
                    shift_w_mantissa_bit = w_mantissa_bit
                
                # print(f"CUR w E{self.w_bit - 1 - shift_w_mantissa_bit}M{shift_w_mantissa_bit}")
                w_sim = self.weight.view(self.n_V,self.crb_rows,self.n_H,self.crb_cols)
                w,cur_w_scale = self.get_w_scale(w_sim, bits = self.w_bit, mantissa_bit= shift_w_mantissa_bit, bias= self.w_intervals[w_mantissa_bit])
                w_sim = (w/cur_w_scale).round_().mul_(cur_w_scale)
                w_sim = w_sim.view(-1,self.in_features)
                bias_sim = self.bias if self.bias is not None else None
                out_sim = F.linear(x_sim, w_sim, bias_sim) # shape: B,*,oc
                similarity = self._get_similarity(raw_out, out_sim, self.metric, raw_grad) #B,*,oc
                similarity = torch.mean(similarity) # shape: 1
                similarities.append(similarity)
            similarities = torch.tensor(similarities)
            batch_similarities.append(similarities)
        batch_similarities = torch.vstack(batch_similarities)
        best_mantissa_bit = batch_similarities.sum(dim=0, keepdim=True).argmax(dim=1).item()

        if self.w_bit == 8:
            self.w_mantissa_bit = torch.tensor(best_mantissa_bit + 2).to(self.weight.device)
            self.w_exponent_bit = torch.tensor(self.w_bit - 1 - self.w_mantissa_bit).to(self.weight.device)    
        
        else:
            self.w_mantissa_bit = torch.tensor(best_mantissa_bit).to(self.weight.device)
            self.w_exponent_bit = torch.tensor(self.w_bit - 1 - best_mantissa_bit).to(self.weight.device)  
        
        self.w_interval = self.w_intervals[best_mantissa_bit]
        # print(f"search result E{self.w_exponent_bit}M{self.w_mantissa_bit}")
        # print("finish searching fp format for linear weight")

    def _search_best_a_interval(self, input_interval_candidates):
        
        for man in range(input_interval_candidates.shape[0]):
            
            tmp_a_interval = self.a_intervals[man].unsqueeze(-1) # shape: n_a,1,1

            for a in range(tmp_a_interval.shape[0]): # the whole tensor only has one scaling factor
                batch_similarities = [] # similarities, need to concatenate and calculate sum (equivalent to mean with argmax)
                for b_st in range(0, self.calib_size, self.calib_batch_size):
                    b_ed = min(self.calib_size, b_st + self.calib_batch_size)
                    x = self.raw_input[b_st:b_ed].to(self.weight.device)
                    raw_out_expanded = self.raw_out[b_st:b_ed].to(self.weight.device).unsqueeze(-2) # shape: b,*,1,oc
                    raw_grad = self.raw_grad
                    similarities = []
                    for p_st in range(0,self.eq_n,self.parallel_eq_n):
                        p_ed = min(self.eq_n, p_st+self.parallel_eq_n)
                        cur_a_interval = tmp_a_interval.repeat(1,1,p_ed-p_st) # shape: n_a,1,parallel_eq_n
                        cur_a_interval[a:a+1,:,:] = input_interval_candidates[man][a:a+1,:,p_st:p_ed]
                        # quantize weight and bias 
                        w_sim, bias_sim = self.quant_weight_bias()
                        # quantize input
                        x_sim=torch.cat(torch.chunk(x.unsqueeze(-2), chunks=self.n_a, dim=-1), dim=-2).unsqueeze(-1)
                        cur_a_bias = self.a_biases[man].unsqueeze(-1)
                        
                        cur_a_interval_zero_point = torch.round(cur_a_interval)
                        # print(f"cur_a_interval_zero_point {cur_a_interval_zero_point.shape}")
                        # print(f"cur_a_bias {cur_a_bias.shape}")
                        if self.a_bit == 8:
                            # print(f"CUR a E{self.a_bit - 1 - man -2}M{man+2}")
                            cur_a, cur_a_scale = self.get_scale(x_sim, bits = self.a_bit, mantissa_bit= man+2, bias= cur_a_bias,tensor_scale= cur_a_interval,tensor_scale_zero_point=cur_a_interval_zero_point)
                        else:
                            cur_a, cur_a_scale = self.get_scale(x_sim, bits = self.a_bit, mantissa_bit= man, bias= cur_a_bias,tensor_scale= cur_a_interval,tensor_scale_zero_point=cur_a_interval_zero_point)

                        x_sim=(cur_a/(cur_a_scale)).round_()*(cur_a_scale) # shape: b,*,n_a,crb_acts,parallel_eq_n
                        # print(f"unique a values{torch.unique(x_sim[0]).shape[0]}")
                        x_sim = x_sim.permute(*list(range(len(x_sim.shape)-3)),-1,-3,-2).reshape(*x.shape[:-1],p_ed-p_st,x.shape[-1]) # shape: b,*,parallel_eq_n,ic
                        # print(f"x_sim {x_sim.shape}")
                        # calculate similarity and store them
                        out_sim = F.linear(x_sim, w_sim, bias_sim) # shape: b,*,parallel_eq_n,oc
                        if self.metric != "pearson":
                            similarity = self._get_similarity(raw_out_expanded, out_sim, self.metric, raw_grad) # shape: b,*,parallel_eq_n
                            if len(similarity.shape) > 2:
                                similarity = torch.mean(similarity, dim=list(range(1,len(similarity.shape)-1))) # shape: b, parallel_eq_n
                        else:
                            similarity = self._get_pearson_a(raw_out_expanded, out_sim)
                        similarity = torch.sum(similarity, dim=0, keepdim=True) # shape: 1, parallel_eq_n
                        
                        similarities.append(similarity)
                    # store best input interval and store in tmp_a_interval
                    similarities = torch.cat(similarities, dim=1) # shape: 1, eq_n
                    batch_similarities.append(similarities)
                batch_similarities = torch.cat(batch_similarities, dim=0).sum(dim=0, keepdim=False) # shape: eq_n
                # print(f"linear similarity {batch_similarities.sum()}")
                a_best_index = batch_similarities.argmax(dim=0, keepdim=True).reshape(1,1,-1)
                # a_best_index = batch_similarities.argmax(dim=0, keepdim=True)
                # print(f"a_best_index {a_best_index.shape}")
                # print(f"input_interval_candidates[man] {input_interval_candidates[man].shape}")
                tmp_a_interval[a:a+1,:,:] = torch.gather(input_interval_candidates[man][a:a+1,:,:],dim=2,index=a_best_index)
                
            self.a_intervals[man] = tmp_a_interval.squeeze(-1)
            self.a_intervals_zero_point[man] = torch.round(self.a_intervals[man])

    def _search_best_a_format(self):
        
        batch_similarities = [] # similarities, need to concatenate and calculate sum (equivalent to mean with argmax)

        # format candidate
        if self.a_bit == 8:
            a_mantissa_bits_candidate = [i for i in range(self.a_bit-3)]
        else:
            a_mantissa_bits_candidate = [i for i in range(self.a_bit-1)]
        # quantize input
        w_sim, bias_sim = self.quant_weight_bias()
        # print(f"before search linear activation E{self.a_exponent_bit}M{self.a_mantissa_bit}")
        for b_st in range(0, self.calib_size, self.calib_batch_size):
            b_ed = min(self.calib_size, b_st + self.calib_batch_size)
            x = self.raw_input[b_st:b_ed].to(self.weight.device)
            raw_out = self.raw_out[b_st:b_ed].to(self.weight.device) # shape: b,*,oc
            raw_grad = self.raw_grad
            similarities = []
            
            for a_mantissa_bit in a_mantissa_bits_candidate:
                if self.a_bit == 8:
                    shift_a_mantissa_bit = a_mantissa_bit + 2
                else:
                    shift_a_mantissa_bit = a_mantissa_bit
                
                x_sim = torch.cat(torch.chunk(x.unsqueeze(-2), chunks=self.n_a, dim=-1), dim=-2)

                cur_a_bias = self.a_biases[a_mantissa_bit]
                cur_a_interval = self.a_intervals[a_mantissa_bit]
                cur_a_interval_zero_point = self.a_intervals_zero_point[a_mantissa_bit]
                cur_a, cur_a_scale = self.get_scale(x_sim, bits = self.a_bit, mantissa_bit= shift_a_mantissa_bit, bias= cur_a_bias,tensor_scale= cur_a_interval,tensor_scale_zero_point=cur_a_interval_zero_point)
                
                x_sim=(cur_a/(cur_a_scale)).round_()*(cur_a_scale) # shape: B,*,n_a,crb_acts
                if len(x.shape) == 3:
                    x_sim = x_sim.view(x.shape[0],x.shape[1],x.shape[2])
                else:
                    x_sim = x_sim.view(x.shape[0],1,x.shape[1])
                out_sim = F.linear(x_sim, w_sim, bias_sim) # shape: B,*,oc 
                if len(raw_out.shape) == 2:
                    out_sim = out_sim.view(raw_out.shape[0],raw_out.shape[1])
                similarity = self._get_similarity(raw_out, out_sim, self.metric, raw_grad) #B,*,oc
                similarity = torch.mean(similarity)
                similarities.append(similarity)
            similarities = torch.tensor(similarities)
            batch_similarities.append(similarities)
                
        batch_similarities = torch.vstack(batch_similarities)
        best_mantissa_bit = batch_similarities.sum(dim=0, keepdim=True).argmax(dim=1).item()

        if self.a_bit == 8:
            self.a_mantissa_bit = torch.tensor(best_mantissa_bit + 2).to(self.weight.device)
            self.a_exponent_bit = torch.tensor(self.a_bit - 1 - best_mantissa_bit).to(self.weight.device)    
        
        else:
            self.a_mantissa_bit = torch.tensor(best_mantissa_bit).to(self.weight.device)
            self.a_exponent_bit = torch.tensor(self.a_bit - 1 - best_mantissa_bit).to(self.weight.device)    

        self.a_interval = self.a_intervals[best_mantissa_bit]
        self.a_interval_zero_point = self.a_intervals_zero_point[best_mantissa_bit]
        self.a_bias = self.a_biases[best_mantissa_bit]
        # print(f"search result linear activation E{self.a_exponent_bit}M{self.a_mantissa_bit}")
        # print("finish searching fp format for linear activation")

    def calibration_step2(self):
        """
        Only use cached raw inputs/outs/grads
        """
        self._initialize_calib_parameters()
        self._initialize_intervals()

        # prepare weight intervals and similarities
        weight_interval_candidates = []
        if self.w_bit == 8:
            for m in range(self.w_bit-3):
                weight_interval_candidate = torch.tensor([self.eq_alpha + i*(self.eq_beta - self.eq_alpha)/self.eq_n for i in range(self.eq_n + 1)]).to(self.weight.device).view(-1,1,1,1,1) * self.w_intervals[m].unsqueeze(0)
                weight_interval_candidates.append(weight_interval_candidate.unsqueeze(0)) # shape: num_man_options,eq_n,n_V,1,n_H,1
        else:
            for m in range(self.w_bit-1):
                weight_interval_candidate = torch.tensor([self.eq_alpha + i*(self.eq_beta - self.eq_alpha)/self.eq_n for i in range(self.eq_n + 1)]).to(self.weight.device).view(-1,1,1,1,1) * self.w_intervals[m].unsqueeze(0)
                weight_interval_candidates.append(weight_interval_candidate.unsqueeze(0)) # shape: num_man_options,eq_n,n_V,1,n_H,1
        weight_interval_candidates = torch.vstack(weight_interval_candidates)

        input_interval_candidates = []
        if self.a_bit == 8:
            for m in range(self.a_bit-3): 
                input_interval_candidate = torch.tensor([self.eq_alpha + i*(self.eq_beta - self.eq_alpha)/self.eq_n for i in range(self.eq_n + 1)]).to(self.weight.device).view(1,1,-1) * self.a_intervals[m].unsqueeze(-1)
                input_interval_candidates.append(input_interval_candidate.unsqueeze(0)) # shape: n_a,1,eq_n
            
        else:
            for m in range(self.a_bit-1): 
                input_interval_candidate = torch.tensor([self.eq_alpha + i*(self.eq_beta - self.eq_alpha)/self.eq_n for i in range(self.eq_n + 1)]).to(self.weight.device).view(1,1,-1) * self.a_intervals[m].unsqueeze(-1)
                input_interval_candidates.append(input_interval_candidate.unsqueeze(0)) # shape: n_a,1,eq_n
        input_interval_candidates = torch.vstack(input_interval_candidates)
        
        
        for e in range(self.search_round):
            # search for best weight interval
            self._search_best_w_interval(weight_interval_candidates)
            # search for best input interval
            self._search_best_a_interval(input_interval_candidates)
            # search for best weight format
            self._search_best_w_format()
            # search for best input format
            self._search_best_a_format()

        print(f"final w format E{self.w_exponent_bit}M{self.w_mantissa_bit}")
        # print(f"final self.w_interval {self.w_interval}")
        # print(f"final self.w_interval_zero_point {self.w_interval_zero_point}")
        print(f"final a format E{self.a_exponent_bit}M{self.a_mantissa_bit}")
        # print(f"final self.a_interval {self.a_interval}")
        # print(f"final self.a_interval_zero_point {self.a_interval_zero_point}")
        self.calibrated = True
        # self._bias_correction_quant_forward(self.raw_input.to(self.weight.device)) # debugging
        del self.raw_input, self.raw_out, self.raw_grad
        return None




