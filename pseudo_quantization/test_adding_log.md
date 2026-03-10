• 需要改，且主要不是 entry.py 评测逻辑本身，而是“MoE 模型接入与量化替换策略”。

  你当前代码里对 Pangu(MoE) 的风险点：

  - entry.py:51 / entry.py:56 没开 trust_remote_code，很多 Pangu 类会加载失败。
  - quantize/pre_quant.py:26:26 的 get_blocks 只支持白名单模型类，Pangu 自定义类很可能直接
    NotImplementedError。
  - quantize/quantizer.py:17:17 当前会量化所有 nn.Linear，MoE 的 router/gate 往往不该默认量
    化（会导致路由漂移，精度波动大）。

  可直接改的最小代码（不增加新参数）：

  # quantize/pre_quant.py
  def get_blocks(model):
      # 先走通用结构，覆盖大多数HF/remote_code CausalLM（含很多MoE）
      if hasattr(model, "model") and hasattr(model.model, "layers"):
          return model.model.layers
      if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
          return model.transformer.h

      # 再保留原来的类型分支（可继续保留）
      ...

  # quantize/quantizer.py
  def _is_router_linear(name: str, module: nn.Linear, model: nn.Module) -> bool:
      cfg = getattr(model, "config", None)
      n_exp = None if cfg is None else (
          getattr(cfg, "num_local_experts", None)
          or getattr(cfg, "n_routed_experts", None)
          or getattr(cfg, "num_experts", None)
      )
      lname = name.lower()
      if "gate_proj" in lname:   # LLaMA MLP gate_proj 不应被当 router
          return False
      if n_exp is not None and module.out_features == n_exp:
          return True
      return ("router" in lname) or ("gate" in lname and "expert" in lname)

  @torch.no_grad()
  def make_quant_linear(model: nn.Module, quant_config: QuantConfig):
      ...
      for name, module in named_linears.items():
          if _is_router_linear(name, module, model):
              continue
          ...

  # entry.py
  from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer

  def build_model_and_enc(model_path):
      config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
      config.use_cache = False
      is_moe = any(hasattr(config, k) for k in ["num_local_experts", "n_routed_experts",
  "num_experts"])
      device_map = "auto" if is_moe else None

      model = AutoModelForCausalLM.from_pretrained(
          model_path,
          config=config,
          torch_dtype=torch.float16,
          low_cpu_mem_usage=True,
          device_map=device_map,
          trust_remote_code=True,
      )
      tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True,
  use_fast=False)
      if tokenizer.pad_token is None:
          tokenizer.pad_token = tokenizer.eos_token

      if device_map is None:
          model = model.to("cuda")

      pseudo_quantize_model(model)
      return model, tokenizer

  # main里
  model, tokenizer = build_model_and_enc(args.model_path)
  ...
  lm_eval_model = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=args.batch_size)

  结论：

  - 如果你的 Pangu 是标准 HF 架构且类恰好被 get_blocks 白名单覆盖，可能“不改也能跑”。
  - 但按你这版代码，更大概率需要上述改动，尤其是 get_blocks 通用化 + router 跳过量化。
  - 这些改动都不需要新增 CLI 参数，仍可只改 model/task/w_bit/a_bit 使用。

  • 已实现你要的 5 个任务评测接入代码（不改 entry.py）：

  - 更新任务分发与最小参数接口：inference.py
  - 新增 LiveCodeBench V6：livecodebench_v6.py
  - 新增 SuperGPQA：supergpqa.py
  - 新增 IF-Eval：ifeval.py
  - 新增 BFCL-V3：bfcl_v3.py
  - 脚本改成只需 task/model/w_bit/a_bit：reasoning.sh


• 可以，现阶段先把测评跑起来是可行的。我已经把入口补齐了这 5 个任务：

  - AIME-2025
  - SuperGPQA
  - IF-Eval
  - LiveCodeBench-V6（LiveCodeBench 也会自动走 V6）
  - BFCL-V3

  修改位置：

  - inference.py:32
  - inference.py:128

  另外把 --quant_method 默认设成了 m2xfp，避免忘传参数导致路径拼接报错。

  可直接运行（在 .../pseudo_quantization 目录下）：

  python -m mxq.evaluation.inference --model OpenPangu/OpenPangu1B --dataset AIME-2025
  python -m mxq.evaluation.inference --model OpenPangu/OpenPangu1B --dataset SuperGPQA
  python -m mxq.evaluation.inference --model OpenPangu/OpenPangu1B --dataset IF-Eval
  python -m mxq.evaluation.inference --model OpenPangu/OpenPangu1B --dataset LiveCodeBench-V6
  python -m mxq.evaluation.inference --model OpenPangu/OpenPangu1B --dataset BFCL-V3
