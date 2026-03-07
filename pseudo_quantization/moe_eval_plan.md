  ## Summary

  重构目标是让 mxq/entry.py 成为 Mini-Challenge 的主入口：entry.py 负责加载 BF16 模型并按现有量化框架构造 W4A4 模型，mxq.evaluation 负责统一执行
  下游任务评测与汇总。
  实现上采用方案 1：entry.py 在同一进程内直接调用 evaluation，把量化后的 model/tokenizer 传入评测后端，不再要求 evaluation 只能按模型路径走
  vLLMModelConfig。

  ## Key Changes

  1. 重构 mxq.evaluation 为“后端无关”的评测层

  - 保留现有任务定义与结果汇总职责，但把“模型加载/推理后端”从任务与 CLI 中拆开。
  - 新增一个 Hugging Face in-memory backend，输入为：
      - model
      - tokenizer
      - generation 参数
      - batch size / max length / seed
  - 现有 main_vllm.py 保留为路径加载型后端，不作为 W4A4 challenge 主链路。
  - inference.py 改成统一调度层：根据输入选择 vllm 后端或 hf_in_memory 后端。

  2. 让 entry.py 直接驱动正式评测

  - entry.py 新增 challenge/eval 模式，支持两类运行：
      - BF16 baseline：仅加载模型，不做量化
      - W4A4：按现有 QuantConfig + make_quant_linear 构造量化模型
  - 构造完模型后，直接调用 mxq.evaluation 的公共评测函数，而不是走 lm_eval.simple_evaluate。
  - 保留现有 wikitext/c4/ptb 和传统 lm_eval 测试路径，不影响日常回归。
  - 新增参数区分：
      - --eval_suite 或 --challenge_eval
      - --run_name bf16|w4a4
      - challenge 任务选择与输出目录
  - 量化配置继续沿用现有 --w_bit --w_mode --a_bit --a_mode --group_size，不在评测层重复定义。

  3. 补齐 Mini-Challenge 任务层

  - 保留并校验现有 AIME-2025、LiveCodeBench 任务。
  - 新增 SuperGPQA、IF-Eval、BFCL-V3 的 task adapter，统一放在 mxq.evaluation.tasks。
  - 每个 task adapter 只负责：
      - 数据集加载
      - prompt 构造
      - 预测后处理
      - 任务指标计算
  - 任务接口统一输出 task-level score，便于后续聚合。
  - 对 LiveCodeBench 固定到 challenge 要求版本 V6，不要继续使用当前“自动枚举全部 config”的默认行为作为正式结果来源。

  4. 新增 challenge 聚合与判定

  - 新增统一聚合模块，读取 BF16 与 W4A4 两次运行结果，计算：
      - 每任务 score
      - 平均 baseline score
      - 平均 W4A4 score
      - mean absolute percentage precision loss
  - 固化判定规则：
      - 单任务 loss = abs((bf16_score - w4a4_score) / bf16_score)
      - 最终指标 = 5 个任务单任务 loss 的平均值
      - 同时输出 challenge 判定：avg_w4a4 >= 0.99 * avg_bf16
  - 聚合器对任务缺失、分母为 0、结果文件不完整直接报错，不隐式跳过。

  ## Public Interfaces

  - mxq.entry
      - 新增正式评测模式入口，推荐形态：
          - python -m mxq.entry --model_path ... --challenge_eval --run_name bf16
          - python -m mxq.entry --model_path ... --challenge_eval --run_name w4a4 --w_bit 4 --a_bit 4 ...
      - 在 challenge 模式下调用 mxq.evaluation 公共函数，而不是 lm_eval.simple_evaluate
  - mxq.evaluation
      - 新增公共函数，形如 evaluate_model(model, tokenizer, datasets, generation_config, output_dir, ...)
      - 新增聚合入口，形如 aggregate_challenge_results(bf16_dir, w4a4_dir, output_path)
  - mxq.evaluation.main_vllm
      - 保留，用于路径加载模型的独立评测，不作为 W4A4 主路径
  - mxq.evaluation.inference
      - 调整为统一入口层，支持 in-memory HF backend 与现有 vLLM backend

  ## Test Plan

  1. 主链路测试

  - entry.py 在 BF16 challenge 模式下能完整跑通至少 1 个样本的单任务评测。
  - entry.py 在 W4A4 模式下完成量化后，能对同一任务跑出结果。
  - BF16 与 W4A4 都走同一套 task/metric 代码，结果目录结构一致。

  2. 后端一致性测试

  - 对已有任务 AIME-2025，对比旧 evaluation + vllm 与新 entry -> evaluation(hf_in_memory) 在小样本上的输出格式与指标字段一致。
  - 确认 in-memory backend 不会绕过量化层，实际调用的是量化后的 nn.Linear 替换结果。

  3. 聚合器测试

  - 用构造结果验证 loss 公式、平均分、pass/fail 判定。
  - 缺失某个任务结果时报错。
  - BF16 task score 为 0 时给出显式异常，避免除零后产生无意义结论。

  4. 回归测试

  - entry.py 原有 wikitext/c4/ptb 路径保持可用。
  - entry.py 原有普通 lm_eval 路径保持可用。
  - 现有 mxq.evaluation.inference 的路径加载型评测不因重构失效。

  ## Assumptions

  - W4A4 模型是运行时量化后的内存模型，不要求导出为可重载 checkpoint。
  - 正式 challenge 结果必须让 BF16 与 W4A4 共用同一套任务实现与打分逻辑。
  - 当前 evaluation 对任务层的价值保留，但其 vLLM-only 绑定需要被拆开；W4A4 主链路默认改用 HF in-memory backend。
  - entry.py 继续承担模型构造职责，evaluation 不接管量化逻辑。
