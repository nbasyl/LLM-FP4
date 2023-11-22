
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import math

class FPMinMaxQuantEmbedding(nn.Embedding):
    def __init__(self,
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: int,
        mode = "raw",
        bit = 8,
        bias_bit = None,
        bias_correction=False,
        exponent_bit = 4, 
        metric=None, search_round=None, eq_alpha=None, eq_beta=None, eq_n=None, parallel_eq_n=None, n_H=None, n_V=None):
        super().__init__(num_embeddings,embedding_dim, padding_idx)
        self.n_calibration_step=2
        self.mode = mode
        self.bit = bit
        assert bias_bit is None,"No support bias bit now"
        assert exponent_bit < (bit - 1)

        self.register_buffer('exponent_bit', torch.tensor(exponent_bit))
        self.register_buffer('mantissa_bit',torch.tensor(bit - 1 - exponent_bit))
        self.register_buffer('interval', None)
        self.default_bias = 2 ** (self.exponent_bit - 1)

        
    def get_maxval_from_bias(self):
        if self.interval != None:
            return (2 - 2 ** (-self.mantissa_bit)) * 2 ** (
                2**self.exponent_bit - 1 - self.interval
            )
        else:
            raise AssertionError
    
    def get_log_scale(self, x):
        
        maxval = self.get_maxval_from_bias()
        bias = self.interval if self.interval != None else self.default_bias
        minval = -maxval
        a = torch.min(torch.max(x, minval), maxval)
        log_scales = torch.clamp((torch.floor(torch.log2(torch.abs(a)) + bias)), 1.0)
        return a, 2.0 ** (log_scales - self.mantissa_bit - bias.float())
        

    def get_scale(self, input, bits, mantissa_bit, bias):
        
        M = mantissa_bit
        E = bits - 1 - M
        bias = bias.float()
        maxval = (2 - 2 ** (-M)) * 2 ** (
                2**E - 1 - bias
            )
        minval = -maxval

        input = torch.min(torch.max(input, minval), maxval)
  
        input_log_scales = torch.clamp((torch.floor(torch.log2(torch.abs(input)) + bias)), 1.0)

        return input, 2.0 ** (input_log_scales - M - bias)


    def forward(self, x):
        if self.mode=='raw':
            out=F.embedding(x, self.weight)
        elif self.mode=="quant_forward":
            out=self.quant_forward(x)
        elif self.mode=="calibration_step1":
            print("embedding calibraion doesn't require step 1")
        elif self.mode=="calibration_step2":
            self.calibration_step2()
        else:
            raise NotImplementedError
        return out
    
    def quant_embed(self):
        w, w_scale = self.get_log_scale( self.weight)
        w=(w/w_scale).round_()
        w_sim=w.mul_(w_scale)
        return w_sim
    
    def quant_forward(self,x):
        assert self.calibrated is not None,f"You should run calibrate_forward before run quant_forward for {self}"
        embed_sim =self.quant_embed()
        out=F.embedding(x, embed_sim)
        return out
    
    def _initialize_intervals_eval(self):
        maxval = self.weight.data.abs().max()
        self.interval = 2**self.exponent_bit - torch.log2(maxval) + math.log2(2 - 2 ** (-self.mantissa_bit)) - 1
        self.calibrated=True

    def calibration_step2(self):
        # step2: search for the best bias for w and a of each layer
        maxval = self.weight.data.abs().max()
        self.interval = 2**self.exponent_bit - torch.log2(maxval) + math.log2(2 - 2 ** (-self.mantissa_bit)) - 1
        self.calibrated=True

class FPPTQSLQuantEmbedding(FPMinMaxQuantEmbedding):
    """
    Floating Point PTQSL on linear modules.
    """
    def __init__(self, 
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: int,
        mode = "raw",
        bit = 8,
        bias_bit = None,
        bias_correction = False,
        exponent_bit = 4,
        metric="L2_norm", search_round=1, eq_alpha=0, eq_beta=1, eq_n=100, parallel_eq_n=1, n_H=1, n_V=1):
        super().__init__(num_embeddings, embedding_dim, padding_idx, mode=mode, bit=bit, bias_bit=bias_bit, exponent_bit=exponent_bit)
        self.metric = metric
        self.search_round = search_round
        self.eq_alpha = eq_alpha
        self.eq_beta = eq_beta
        self.eq_n = int(eq_n)
        self.n_V = n_V
        self.n_H = n_H
        self.crb_rows = num_embeddings // n_V
        self.crb_cols = embedding_dim // n_H
        self.parallel_eq_n = parallel_eq_n

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
    
    def quant_embed(self):
        # self.w_interval shape: n_V, 1, n_H, 1
        w_sim = self.weight.view(self.n_V,self.crb_rows,self.n_H,self.crb_cols)
        w, w_scale = self.get_log_scale(w_sim)
        w_sim = (w / w_scale).round_()
        w_sim = w_sim.mul_(w_scale).view(self.num_embeddings,self.embedding_dim)

        return w_sim
    
    def _search_best_interval(self, interval_candidates):
        """
        Modularization of searching best weight intervals
        """
        tmp_w_interval = self.w_interval.unsqueeze(0) # shape: 1,n_V,1,n_H,1
        for h in range(self.n_H):
            similarities = []
            for p_st in range(0,self.eq_n,self.parallel_eq_n):
                p_ed = min(self.eq_n, p_st+self.parallel_eq_n)
                cur_w_interval = tmp_w_interval.repeat(p_ed-p_st,1,1,1,1)
                cur_w_interval[:,:,:,h:h+1,:] = interval_candidates[p_st:p_ed,:,:,h:h+1,:]
                # quantize weight
                w_sim = self.weight.view(self.n_V,self.crb_rows,self.n_H,self.crb_cols).unsqueeze(0) # shape: 1,n_V,crb_rows,n_H,crb_cols
                
                w, cur_w_scale = self.get_scale(w_sim, bits = self.w_bit, mantissa_bit= self.w_mantissa_bit, bias= cur_w_interval)

                w_sim = (w/cur_w_scale).round_().mul_(cur_w_scale) # shape: parallel_eq_n,n_V,crb_rows,n_H,crb_cols
                w_sim = w_sim.view(-1,self.num_embeddings,self.embedding_dim) # shape: parallel_eq_n*oc,ic
                

                
                similarity = self._get_similarity(self.weight.unsqueeze(-1), w_sim, self.metric) # shape: B,*,parallel_eq_n,n_V
                similarity = torch.mean(similarity, dim=list(range(len(similarity.shape)-2))) # shape: parallel_eq_n, n_V
                similarities.append(similarity)
            # store best weight interval of h into tmp_w_interval
            similarities = torch.cat(similarities, dim=0) # shape: eq_n, n_V
            h_best_index = similarities.argmax(dim=0).reshape(1,-1,1,1,1) # shape: 1,n_V,1,1,1
            tmp_w_interval[:,:,:,h:h+1,:] = torch.gather(interval_candidates[:,:,:,h:h+1,:],dim=0,index=h_best_index)
        self.w_interval = tmp_w_interval.squeeze(dim=0)
    
    def _initialize_intervals(self):
        
        self.n_V = self.num_embeddings
        self.crb_rows = self.num_embeddings // self.n_V
        maxval = self.weight.view(self.n_V,self.crb_rows,self.n_H,self.crb_cols).abs().amax([1,3],keepdim=True)
        self.interval=(2**self.exponent_bit - torch.log2(maxval) + math.log2(2 - 2 ** (-self.mantissa_bit)) - 1)
   
    def calibration_step2(self):
        # initialize intervals with minmax intervals
        self._initialize_intervals()

        # prepare weight intervals and similarities
        interval_candidates = torch.tensor([self.eq_alpha + i*(self.eq_beta - self.eq_alpha)/self.eq_n for i in range(self.eq_n + 1)]).to(self.weight.device).view(-1,1,1,1,1) * self.w_interval.unsqueeze(0) # shape: eq_n,n_V,1,n_H,1
        for e in range(self.search_round):
            # search for best weight interval
            self._search_best_interval(interval_candidates)

        self.calibrated = True
    
class FPPTQSLQuantEmbedding_fpq_baseline(FPPTQSLQuantEmbedding):
    def __init__(self, 
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: int,
        mode = "raw",
        bit = 8,
        exponent_bit = 4,
        bias_bit = None,
        bias_correction = False,
        metric="L2_norm", search_round=1, eq_alpha=0, eq_beta=1, eq_n=100, parallel_eq_n=1, n_H=1, n_V=1):
        super().__init__(num_embeddings, embedding_dim, padding_idx, mode=mode, bit=bit, exponent_bit= exponent_bit, bias_bit=bias_bit, bias_correction=bias_correction, metric=metric, search_round=search_round, eq_alpha=eq_alpha, eq_beta=eq_beta, eq_n=eq_n, parallel_eq_n=parallel_eq_n, n_H=n_H, n_V=n_V)
        self.maxval = None
        self.intervals = None

    def _initialize_intervals_eval(self):

        self.n_V = self.num_embeddings
        self.crb_rows = self.num_embeddings // self.n_V
        maxval = self.weight.view(self.n_V, self.crb_rows,self.n_H,self.crb_cols).abs().amax([1,3],keepdim=True)
        self.maxval = maxval
        self.interval=(2**self.exponent_bit - torch.log2(maxval) + math.log2(2 - 2 ** (-self.mantissa_bit)) - 1)
        self.calibrated = True

    def _initialize_intervals(self):

        self.n_V = self.num_embeddings
        self.crb_rows = self.num_embeddings // self.n_V
        maxval = self.weight.view(self.n_V, self.crb_rows,self.n_H,self.crb_cols).abs().amax([1,3],keepdim=True)
        self.maxval = maxval
        self.interval=(2**self.exponent_bit - torch.log2(maxval) + math.log2(2 - 2 ** (-self.mantissa_bit)) - 1)
        self.intervals = []
        if self.bit == 8: ## need to constrain the exponent as too big exponent bits will result in overflow
            # E7M0, E6M1, E5M2, E4M3, E3M4, E2M5, E1M6, start with E5M2 as E7M0 and E6M1 usually performs quite bad and results in overflow
            for i in range(self.bit-3):
                M = i + 2
                E = self.bit - 1 - M
                self.intervals.append(2**E - torch.log2(self.maxval) + math.log2(2 - 2 ** (-M)) - 1)

        else:
            for i in range(self.bit-1):
                M = i
                E = self.bit - 1 - M
                self.intervals.append(2**E - torch.log2(self.maxval) + math.log2(2 - 2 ** (-M)) - 1)

    def _get_similarity(self, tensor_raw, tensor_sim, metric=None):
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

    def _search_best_interval(self, interval_candidates):
        
        # print(f"interval_candidates shape {interval_candidates.shape}")
        for man in range(interval_candidates.shape[0]):
            tmp_interval = self.intervals[man].unsqueeze(0) # shape: 1,n_V,1,n_H,1
            for h in range(self.n_H):
                similarities = []
                for p_st in range(0,self.eq_n,self.parallel_eq_n):
                    p_ed = min(self.eq_n, p_st+self.parallel_eq_n)
                    cur_w_interval = tmp_interval.repeat(p_ed-p_st,1,1,1,1)
                    cur_w_interval[:,:,:,h:h+1,:] = interval_candidates[man][p_st:p_ed,:,:,h:h+1,:]
                    # quantize weight and bias 
                    w_sim = self.weight.view(self.n_V,self.crb_rows,self.n_H,self.crb_cols).unsqueeze(0) # shape: 1,n_V,crb_rows,n_H,crb_cols
                    
                    if self.bit >= 8:
                        w, cur_w_scale = self.get_scale(w_sim, bits = self.bit, mantissa_bit= man+2, bias= cur_w_interval)
                    else:
                        w, cur_w_scale = self.get_scale(w_sim, bits = self.bit, mantissa_bit= man, bias= cur_w_interval)

                    w_sim = (w/cur_w_scale).round_().mul_(cur_w_scale) # shape: parallel_eq_n,n_V,crb_rows,n_H,crb_cols
                    w_sim = w_sim.view(-1,self.num_embeddings,self.embedding_dim) # shape: parallel_eq_n*oc,ic
                    

                    similarity = self._get_similarity(self.weight.unsqueeze(0), w_sim, self.metric) # shape: B,*,parallel_eq_n,n_V
                    if self.n_V == 1:
                        similarity = similarity.sum(dim=1, keepdim=True)
                
                    similarities.append(similarity)
                # store best weight interval of h into tmp_interval
                similarities = torch.cat(similarities, dim=0) # shape: eq_n, n_V
                h_best_index = similarities.argmax(dim=0).reshape(1,-1,1,1,1) # shape: 1,n_V,1,1,1
                tmp_interval[:,:,:,h:h+1,:] = torch.gather(interval_candidates[man][:,:,:,h:h+1,:],dim=0,index=h_best_index)
            self.intervals[man] = tmp_interval.squeeze(dim=0)

    def _search_best_format(self):
        
        # print(f"before search linear weight E{self.w_exponent_bit}M{self.w_mantissa_bit}")
        
        # format candidate
        if self.bit >= 8:
            mantissa_bits_candidate = [i for i in range(self.bit-3)]
        else:
            mantissa_bits_candidate = [i for i in range(self.bit-1)]
        
        similarities = []
        for mantissa_bit in mantissa_bits_candidate:
            if self.bit >= 8:
                shift_mantissa_bit = mantissa_bit + 2
            else:
                shift_mantissa_bit = mantissa_bit
                
            w_sim = self.weight.view(self.n_V,self.crb_rows,self.n_H,self.crb_cols)
            w, cur_w_scale = self.get_scale(w_sim, bits = self.bit, mantissa_bit= shift_mantissa_bit, bias= self.intervals[mantissa_bit])
 
            w_sim = (w/cur_w_scale)
            
            w_sim = w_sim.round_().mul_(cur_w_scale)

    
            w_sim = w_sim.view(-1,self.num_embeddings,self.embedding_dim)

            similarity = self._get_similarity(self.weight.unsqueeze(0), w_sim, self.metric) #B,*,oc
            similarity = torch.mean(similarity) # shape: 1
            similarities.append(similarity)
        similarities = torch.tensor(similarities)
        best_mantissa_bit = similarities.argmax(dim=0).item()
        
        if self.bit >= 8:
            self.mantissa_bit = torch.tensor(best_mantissa_bit + 2).to(self.weight.device)
            self.exponent_bit = torch.tensor(self.bit - 1 - best_mantissa_bit).to(self.weight.device) 
        
        else:
            self.mantissa_bit = torch.tensor(best_mantissa_bit).to(self.weight.device) 
            self.exponent_bit = torch.tensor(self.bit - 1 - best_mantissa_bit).to(self.weight.device) 
            
        self.interval = self.intervals[best_mantissa_bit]

    def calibration_step2(self):

        self._initialize_intervals()

        # prepare intervals and similarities
        interval_candidates = []
        if self.bit >=8:
            for m in range(self.bit-3): #m 2 ~ 6
                interval_candidate = torch.tensor([self.eq_alpha + i*(self.eq_beta - self.eq_alpha)/self.eq_n for i in range(self.eq_n + 1)]).to(self.weight.device).view(-1,1,1,1,1) * self.intervals[m].unsqueeze(0)
                interval_candidates.append(interval_candidate.unsqueeze(0)) # shape: num_man_options,eq_n,n_V,1,n_H,1
            
        else:
            for m in range(self.bit-1): #m 0 ~ 6
                interval_candidate = torch.tensor([self.eq_alpha + i*(self.eq_beta - self.eq_alpha)/self.eq_n for i in range(self.eq_n + 1)]).to(self.weight.device).view(-1,1,1,1,1) * self.intervals[m].unsqueeze(0)
                interval_candidates.append(interval_candidate.unsqueeze(0)) # shape: num_man_options,eq_n,n_V,1,n_H,1
        interval_candidates = torch.vstack(interval_candidates)

        for e in range(self.search_round):
            # search for best weight interval
            self._search_best_interval(interval_candidates)
            # search for best weight format
            self._search_best_format()

        print(f"search format E{self.exponent_bit}M{self.mantissa_bit}")

        self.calibrated = True
        return None

