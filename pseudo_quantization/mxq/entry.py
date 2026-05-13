from lm_eval import evaluator
from lm_eval.models.huggingface import HFLM
from lm_eval.utils import make_table

from transformers import (
    AutoModelForCausalLM,
    AutoConfig,
)
import torch
import argparse
from mxq.quantize.quant_func import QuantConfig
from mxq.quantize.quantizer import make_quant_linear
# from mxq.quantize.awq.prequant import run_awq
# from mxq.quantize.awq.prequant import apply_awq

import datetime
import tqdm
from torch import nn


def print_time(print_str):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"{timestamp} - {print_str}")


parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str, help="path of the hf model")
parser.add_argument("--batch_size", type=int, default=32, help="batch size")
parser.add_argument("--tasks", default=None, type=str)
parser.add_argument("--num_fewshot", type=int, default=0)
parser.add_argument("--limit_samples", type=int, default=None)

# quantization config
parser.add_argument("--w_bit", type=int, default=16)
parser.add_argument(
        "--w_mode", type=str, choices=["mxfp", "nvfp", "mxem", "mxes", "nvem", "nves", "nvesem2", "nvesm", "nvesm2", "nvesm2_hw", "nvint4", "nvintesm2", "hif4", "hifem", "hifes", "nvgt4"], default=None
)
parser.add_argument("--a_bit", type=int, default=16)
parser.add_argument(
    "--a_mode", type=str, choices=["mxfp", "nvfp", "mxem", "mxes", "nvem", "nves", "nvesem2", "nvesm", "nvesm2", "nvesm2_hw", "nvint4", "nvintesm2", "hif4", "hifem", "hifes", "nvgt4"], default=None
)
parser.add_argument("--group_size", type=int, default=-1)
parser.add_argument("--awq", action="store_true", help="Whether to use AWQ")

args = parser.parse_args()
if args.limit_samples is not None and args.limit_samples <= 0:
    raise ValueError("--limit_samples must be a positive integer")


# build model and tokenizer
def build_model_and_enc(model_path):
    print(f"* Building model {model_path}")

    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    # fp16 to quantized
    # kwargs = {"device_map": "balanced", "torch_dtype": torch.float16}
    # model = AutoModelForCausalLM.from_pretrained(model_path, config=config, **kwargs)
    config.use_cache = False
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        config=config,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map=None,          # ★ 禁止 balanced
        trust_remote_code=True,
    )

    # if args.awq:
    #     from transformers import AutoTokenizer
    #
    #     tokenizer = AutoTokenizer.from_pretrained(
    #         args.model_path, use_fast=False
    #     )
    #     # # if tokenizer.pad_token is None:
    #     # #     tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    #     # model.resize_token_embeddings(len(tokenizer))
    #
    #     q_config = {
    #         "zero_point": True,  # by default True
    #         "q_group_size": args.group_size,  # whether to use group quantization
    #         "quant_mode": args.w_mode,  # quantization mode
    #     }
    #     model.eval().cuda()
    #
    #     print_time("Start AWQ quantization")
    #     awq_results = run_awq(
    #         model,
    #         tokenizer,
    #         w_bit=args.w_bit,
    #         q_config=q_config,
    #         n_samples=128,
    #         seqlen=512,
    #         auto_scale=True,
    #         mse_range=True,
    #         calib_data="pileval",
    #     )
    #     print_time("Finish AWQ quantization")
    #
    #     apply_awq(model, awq_results)
    #
    model = model.to("cuda")

    pseudo_quantize_model(model)
    return model


def pseudo_quantize_model(model):

    print_time("Start pseudo quantize")

    quant_config = QuantConfig(
        w_bit=args.w_bit,
        w_mode=args.w_mode,
        a_bit=args.a_bit,
        a_mode=args.a_mode,
        group_size=args.group_size,
    )
    make_quant_linear(model, quant_config)
    print_time("Finish pseudo quantize")

def main():

    print("\nargs:", args, "\n")

    model = build_model_and_enc(args.model_path)


    if args.tasks is not None:
        if args.tasks in ["wikitext", "c4", "ptb"]:
            # https://github.com/IST-DASLab/gptq/blob/2d65066eeb06a5c9ff5184d8cebdf33662c67faf/llama.py#L206
            from .utils.dataload_utils import get_loaders

            model.seqlen = 2048
            _, testenc = get_loaders(
                args.tasks, model=args.model_path, seqlen=model.seqlen
            )

            testenc = testenc.input_ids.to(model.device)
            nsamples = testenc.numel() // model.seqlen
            if args.limit_samples is not None:
                nsamples = min(nsamples, args.limit_samples)
            model = model.eval()
            nlls = []
            for i in tqdm.tqdm(range(nsamples), desc="evaluating..."):
                batch = testenc[:, (i * model.seqlen) : ((i + 1) * model.seqlen)].to(
                    model.device
                )
                with torch.no_grad():
                    lm_logits = model(batch).logits
                shift_logits = lm_logits[:, :-1, :].contiguous().float()
                shift_labels = testenc[
                    :, (i * model.seqlen) : ((i + 1) * model.seqlen)
                ][:, 1:]
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(
                    shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
                )
                neg_log_likelihood = loss.float() * model.seqlen
                nlls.append(neg_log_likelihood)

            ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * model.seqlen))
            print(ppl.item())

        else:
            # do other evaluations
            # print("no implementation yet")
            lm_eval_model = HFLM(
                pretrained=model,
                batch_size=args.batch_size,
                trust_remote_code=True,
            )
            print_time("Start a task")
            task_names = args.tasks.split(",")

            results = evaluator.simple_evaluate(
                model=lm_eval_model,
                tasks=task_names,
                batch_size=args.batch_size,
                num_fewshot=args.num_fewshot,
                limit=args.limit_samples,
            )
            print_time("Task finish!")
            print(make_table(results))


if __name__ == "__main__":
    main()
