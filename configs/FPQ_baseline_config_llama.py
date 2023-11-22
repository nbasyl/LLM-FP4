from quant_layers.fp_linear import FPPTQSLBatchingQuantLinear_fpq_baseline
from quant_layers.fp_embed import FPPTQSLQuantEmbedding_fpq_baseline


bit = 8
exp_bit = 4
embed_name_list = ["qembedding"]
fc_name_list = [ "qlinear_query", "qlinear_key", "qlinear_value", "qlinear_o","qlinear_gate","qlinear_down","qlinear_up","qlinear_score"]
matmul_name_list = [ "qmatmul_qk", "qmatmul_scorev"]
w_bit = {name: bit for name in fc_name_list}
a_bit = {name: bit for name in fc_name_list}
embed_bit = {name: bit for name in embed_name_list}
A_bit = {name: bit for name in matmul_name_list}
B_bit = {name: bit for name in matmul_name_list}
w_exp_bit = {name: exp_bit for name in fc_name_list}
a_exp_bit = {name: exp_bit for name in fc_name_list}
embed_exp_bit = {name: exp_bit for name in embed_name_list}
A_exp_bit = {name: exp_bit for name in matmul_name_list}
B_exp_bit = {name: exp_bit for name in matmul_name_list}

ptqsl_embedding_kwargs = {
    "metric": "L2_norm",
    "eq_alpha": 0.01,
    "eq_beta": 1.2,
    "eq_n": 100,
    'search_round': 3,
    "n_V": 1,
    "n_H": 1
}
ptqsl_linear_kwargs = {
    "metric": "L2_norm",
    "eq_alpha": 0.01,
    "eq_beta": 1.2,
    "eq_n": 100,
    'search_round': 3,
    "n_V": 1,
    "n_H": 1,
    "n_a": 1,
    "bias_correction":True # Conventionally I'll not add an actual bias correction in linear
}


def get_module(module_type, *args, **kwargs):

    if "embedding" in module_type:
        kwargs.update(ptqsl_embedding_kwargs)
        module= FPPTQSLQuantEmbedding_fpq_baseline(*args,**kwargs,bit= embed_bit[module_type], exponent_bit=embed_exp_bit[module_type], padding_idx=0)


    elif "qlinear" in module_type:
        kwargs.update(ptqsl_linear_kwargs)
        if module_type == "qlinear_score":
            kwargs["n_V"] = 1
            module= FPPTQSLBatchingQuantLinear_fpq_baseline(*args,**kwargs,w_bit=w_bit[module_type],a_bit=a_bit[module_type],w_exponent_bit=w_exp_bit[module_type],a_exponent_bit=a_exp_bit[module_type])
        else:
            module= FPPTQSLBatchingQuantLinear_fpq_baseline(*args,**kwargs,w_bit=w_bit[module_type],a_bit=a_bit[module_type],w_exponent_bit=w_exp_bit[module_type],a_exponent_bit=a_exp_bit[module_type])
            
    return module