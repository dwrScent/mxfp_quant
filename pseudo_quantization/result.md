# Quantization Evaluation Result

- Run id: 20260529_100659
- Updated at: 2026-05-29 10:28:17
- Weight bit: 16
- Activation bit: 16
- Group size: 16
- Default batch size: 32
- BoolQ batch size: 8
- Methods: fp16
- Models: qwen-7b
- Tasks: wikitext, c4, ptb, hellaswag, piqa, winogrande, arc_easy, arc_challenge, boolq
- Log dir: /root/llm-quan/mxfp_quant/pseudo_quantization/logs

## qwen-7b

### PPL

|  | wikitext | c4 | ptb | avg |
| --- | --- | --- | --- | --- |
| fp16 | 7.603 | 10.016 | 12.408 | 10.009 |

### Accuracy

|  | hellaswag | piqa | winogrande | arc_easy | arc_challenge | boolq | avg |
| --- | --- | --- | --- | --- | --- | --- | --- |
| fp16 | 57.35 | 77.15 | 68.59 | 74.07 | 44.03 | 67.86 (8bs) | 64.842 |

### Accuracy Norm

|  | hellaswag | piqa | winogrande | arc_easy | arc_challenge | boolq | avg |
| --- | --- | --- | --- | --- | --- | --- | --- |
| fp16(norm) | 76.80 | 78.13 |  | 71.84 | 45.90 |  | 68.168 |
