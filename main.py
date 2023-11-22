import argparse
import json
import logging
import os

from lm_eval import tasks, evaluator, utils

try:
    import wandb

    has_wandb = True
except ImportError:
    has_wandb = False

logging.getLogger("openai").setLevel(logging.WARNING)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--model_args", default="")
    parser.add_argument("--tasks", default=None, choices=utils.MultiChoice(tasks.ALL_TASKS))
    parser.add_argument("--provide_description", action="store_true")
    parser.add_argument("--num_fewshot", type=int, default=0)
    parser.add_argument("--batch_size", type=str, default=None)
    parser.add_argument("--max_batch_size", type=int, default=None,
                        help="Maximal batch size to try with --batch_size auto")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--output_path", default=None)
    parser.add_argument("--limit", type=float, default=None,
                        help="Limit the number of examples per task. "
                             "If <1, limit is a percentage of the total number of examples.")
    parser.add_argument("--data_sampling", type=float, default=None)
    parser.add_argument("--no_cache", action="store_true")
    parser.add_argument("--decontamination_ngrams_path", default=None)
    parser.add_argument("--description_dict_path", default=None)
    parser.add_argument("--check_integrity", action="store_true")
    parser.add_argument("--write_out", action="store_true", default=False)
    parser.add_argument("--output_base_path", type=str, default=None)
        
    parser.add_argument("--wandb_name", type = str)
    parser.add_argument("--quant_config", type = str, default = 'FPT_config_llama', help="The PTQ method to use, [FPQ_config_llama, FPQ_baseline_config_llama, MinMax_config_llama]")
    parser.add_argument("--qbits", nargs='+', type=int, help="In the format of (Weight bit widths, Activation bit widths, Embedding bit widths, W_exponent_bit, A_exponent_bit, Embedding_exponent_bit)")
    parser.add_argument("--calib_size", type=int, default=32, help="Total number of calibration data sample")
    parser.add_argument("--search_round", type=int, default=3, help="Total number of search round, must be larger than 1")
    parser.add_argument("--search_intervals", nargs='+', type=float, default=(0.01,1.2,100) ,help="In the format of (gamma_1, gamma_2, search_interval)")
    
    parser.add_argument("--only_eval", action="store_true", default=False, help='use only when you have the PTQ param and only want to evaluate')
    parser.add_argument("--ptq_param_path", type=str, default="./q_params/FP4_FPQ.pt")

    return parser.parse_args()


def main():
    args = parse_args()

    print(f"config: {args}")
    
    if has_wandb:

        print("init wandb")
        wandb.init(project=args.wandb_name, config=args)

    assert not args.provide_description  # not implemented

    if args.limit:
        print(
            "WARNING: --limit SHOULD ONLY BE USED FOR TESTING. REAL METRICS SHOULD NOT BE COMPUTED USING LIMIT."
        )

    if args.tasks is None:
        task_names = tasks.ALL_TASKS
    else:
        task_names = utils.pattern_match(args.tasks.split(","), tasks.ALL_TASKS)

    print(f"Selected Tasks: {task_names}")

    description_dict = {}
    if args.description_dict_path:
        with open(args.description_dict_path, "r") as f:
            description_dict = json.load(f)

    if args.only_eval:
        results = evaluator.only_evaluate(
            model=args.model,
            model_args=args.model_args,
            tasks=task_names,
            num_fewshot=args.num_fewshot,
            batch_size=args.batch_size,
            max_batch_size=args.max_batch_size,
            device=args.device,
            no_cache=args.no_cache,
            limit=args.limit,
            description_dict=description_dict,
            decontamination_ngrams_path=args.decontamination_ngrams_path,
            check_integrity=args.check_integrity,
            write_out=args.write_out,
            output_base_path=args.output_base_path,
            qbits = tuple(args.qbits),
            calib_size = 1,
            quant_config = args.quant_config,
            ptq_param_path= args.ptq_param_path
        )

    else:
        results = evaluator.calibrate_evaluate(
            model=args.model,
            model_args=args.model_args,
            tasks=task_names,
            num_fewshot=args.num_fewshot,
            batch_size=args.batch_size,
            max_batch_size=args.max_batch_size,
            device=args.device,
            no_cache=args.no_cache,
            limit=args.limit,
            description_dict=description_dict,
            decontamination_ngrams_path=args.decontamination_ngrams_path,
            check_integrity=args.check_integrity,
            write_out=args.write_out,
            output_base_path=args.output_base_path,
            qbits = tuple(args.qbits),
            calib_size = args.calib_size,
            quant_config = args.quant_config,
            search_round=args.search_round,
            search_intervals=tuple(args.search_intervals)
        )

    dumped = json.dumps(results, indent=2)
    print(dumped)

    if args.output_path:
        os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
        with open(args.output_path, "w") as f:
            f.write(dumped)

    batch_sizes = ",".join(map(str, results["config"]["batch_sizes"]))
    print(
        f"{args.model} ({args.model_args}), limit: {args.limit}, provide_description: {args.provide_description}, "
        f"num_fewshot: {args.num_fewshot}, batch_size: {args.batch_size}{f' ({batch_sizes})' if batch_sizes else ''}"
    )
    print(evaluator.make_table(results))

    if has_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
