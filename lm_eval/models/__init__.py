from . import huggingface


MODEL_REGISTRY = {
    "hf-causal-experimental": huggingface.AutoCausalLM,
    "hf-seq2seq": huggingface.AutoSeq2SeqLM,
}


def get_model(model_name):
    return MODEL_REGISTRY[model_name]
