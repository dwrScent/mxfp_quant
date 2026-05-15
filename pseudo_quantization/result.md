# Quantization Evaluation Result

- Run id: 20260515_105747
- Updated at: 2026-05-15 14:53:33
- Weight bit: 4
- Activation bit: 4
- Group size: 16
- Default batch size: 32
- BoolQ batch size: 8
- Methods: nvfp, nves, nvint4, nvesm2_hw, nvintesm2
- Models: qwen-7b
- Tasks: wikitext, c4, ptb, hellaswag, piqa, winogrande, arc_easy, arc_challenge, boolq
- Log dir: /root/llm-quan/mxfp_quant/pseudo_quantization/logs

## qwen-7b

### PPL

|  | wikitext | c4 | ptb | avg |
| --- | --- | --- | --- | --- |
| nvfp | 8.116 | 10.653 | 13.251 | 10.673 |
| nves | 8.120 | 10.573 | 13.213 | 10.635 |
| nvint4 | 8.367 | 10.825 | 13.429 | 10.874 |
| nvesm2_hw | 8.049 | 10.532 | 13.117 | 10.566 |
| nvintesm2 |  |  |  |  |

### Accuracy

|  | hellaswag | piqa | winogrande | arc_easy | arc_challenge | boolq | avg |
| --- | --- | --- | --- | --- | --- | --- | --- |
| nvfp | 55.93 | 76.61 | 66.54 | 72.98 | 40.70 | 67.55 (8bs) | 63.385 |
| nves | 56.29 | 76.82 | 66.14 | 73.15 | 41.81 | 67.83 (8bs) | 63.673 |
| nvint4 | 55.26 | 76.55 | 66.85 | 74.41 | 43.94 | 66.79 (8bs) | 63.967 |
| nvesm2_hw |  |  |  |  |  |  |  |
| nvintesm2 |  |  |  |  |  |  |  |

### Accuracy Norm

|  | hellaswag | piqa | winogrande | arc_easy | arc_challenge | boolq | avg |
| --- | --- | --- | --- | --- | --- | --- | --- |
| nvfp(norm) | 75.39 | 77.31 |  | 71.21 | 45.05 |  | 67.240 |
| nves(norm) | 75.87 | 77.37 |  | 71.55 | 45.14 |  | 67.483 |
| nvint4(norm) | 74.67 | 77.09 |  | 72.81 | 45.73 |  | 67.575 |
| nvesm2_hw(norm) |  |  |  |  |  |  |  |
| nvintesm2(norm) |  |  |  |  |  |  |  |
