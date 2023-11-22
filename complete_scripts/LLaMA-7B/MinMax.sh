export CUDA_VISIBLE_DEVICES=0
MODEL_ADDR=huggyllama/llama-7b
python main.py --model hf-causal-experimental --model_args pretrained=$MODEL_ADDR,use_accelerate=True \
--tasks arc_challenge,arc_easy,boolq,hellaswag,openbookqa,piqa,winogrande --device cuda --batch_size auto \
--wandb_name '7B_MinMax_FP4' --no_cache --num_fewshot 0 --quant_config 'MinMax_config_llama' --qbits 4 4 4 2 2 2 --calib_size 32
