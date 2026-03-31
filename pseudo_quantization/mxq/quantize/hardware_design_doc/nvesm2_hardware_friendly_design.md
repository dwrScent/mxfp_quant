# nvesm2 Hardware-Friendly Design Notes

## Local code context

- Original `nvesm2` implementation: `quant_func.py` commented block around `get_quant_nvesm2`.
- Current optimized implementation: `quant_func.py` active `get_quant_nvesm2`.
- Current format intent from `reminder.md`: `nvesm2 = extra 2-bit scale(1.xx) with mse search`.

## What the original algorithm is doing

The commented version is effectively:

1. Use NVFP-style hierarchical scaling.
   - Group scale is derived from group `amax / 6`.
   - The group scale is quantized to FP8 E4M3 via a tensor-level `global_scale`.
2. Split each group into subgroups of 8.
3. For each subgroup, test 4 scale-mantissa candidates:
   - `ratio in {1.0, 1.25, 1.5, 1.75}`
4. For each candidate, quantize to FP4 E2M1 and evaluate error.
5. Pick the best ratio per subgroup.

That is a good offline search algorithm, but it is not a good online hardware algorithm because it still needs:

- 4 parallel candidate paths per subgroup
- repeated scale multiply/divide
- 4-way error accumulation and argmin

## What the current active version changes

The active version replaces explicit `MAE/MSE` search with a heuristic score based on normalized quantization error. That reduces arithmetic precision requirements, but hardware cost is still structurally high because it still keeps:

- 4 candidate normalized values
- 4 candidate dequant paths
- 4-way score accumulation
- argmin select

So the main bottleneck is still there: the encoder is a multi-candidate search engine.

## Recommended direction

I would not keep a 4-candidate search in the hardware datapath.

I recommend keeping the **same metadata format** as `nvesm2`:

- base element format: FP4 E2M1
- one block scale in FP8 E4M3
- one subgroup 2-bit mantissa code selecting `{1.0, 1.25, 1.5, 1.75}`

But I would change the **ratio selection algorithm** to a **single-path estimator + small LUT correction**, instead of candidate search.

## Proposed hardware-friendly nvesm2 encoder

### Core idea

For subgroup `g`, define:

- `s_base`: the quantized block scale already produced by the NVFP-style path
- `a1`: largest absolute value in the subgroup
- `a2`: second-largest absolute value in the subgroup
- `r_raw = a1 / (6 * s_base)`

Then quantize `r_raw` directly to the 2-bit mantissa grid:

- `k0 = clamp(round(4 * (r_raw - 1.0)), 0, 3)`
- `ratio0 = 1.0 + 0.25 * k0`

This already removes the 4-way candidate search.

### Outlier-aware correction

Pure `amax` is fragile when one element is a strong outlier. To reduce overscaling, add one small outlier detector:

- `outlier_flag = (a1 > T * a2)` with `T` calibrated offline, e.g. `1.6~2.0`

Then use a tiny LUT:

- input: `(k0, outlier_flag)`
- output: final `k`

A simple starting rule is:

- if `outlier_flag == 0`, keep `k = k0`
- if `outlier_flag == 1`, use `k = max(k0 - 1, 0)`

This is cheap and directly addresses the main failure mode of subgroup-scale methods: one outlier can force the whole subgroup to over-scale.

### Optional percentile variant

If the hardware budget allows a slightly richer reduction tree, use:

- `a_ref = max(a2, beta * a1)` with `beta in [0.75, 0.875]`
- `r_raw = a_ref / (6 * s_base)`

This is more stable than plain `amax`, still much cheaper than 4-path error search, and only requires top-2 reduction.

## Why this is a better fit than the current heuristic

Compared with the current active implementation, this design:

- removes 4-way candidate generation
- removes 4-way dequant evaluation
- removes 4-way score accumulation and argmin
- turns the encoder into:
  - subgroup reduction (`amax`, optionally second max)
  - one scale normalization
  - one 2-bit quantization
  - one tiny correction LUT

This is a much more natural hardware encoder.

## Datapath sketch

### Offline / static weight path

For weights, exact search is acceptable because it is not on the runtime critical path.

Recommendation:

- Keep the original exact search offline for weights.
- Store the chosen 2-bit subgroup code in the checkpoint.
- Hardware only performs decode and GEMM, with no runtime search.

This gives you the best accuracy for weights with zero runtime encoder cost.

### Online / dynamic activation path

For activations, use the single-path estimator described above.

Per subgroup pipeline:

1. Compute subgroup `a1`, optionally `a2`.
2. Read block scale `s_base`.
3. Estimate `r_raw`.
4. Quantize to `k0 in {0,1,2,3}`.
5. Apply one LUT-based correction using outlier flag.
6. Emit subgroup metadata `k`.
7. Quantize elements with `scale = s_base * (1 + k/4)`.

## Implementation details that matter

### 1. Avoid true division in hardware

Do not implement general division for `a1 / (6 * s_base)`.

Use either:

- reciprocal LUT indexed by FP8 scale code, or
- exponent/mantissa split because `s_base` is FP8 E4M3.

Since `s_base` is already quantized, a reciprocal table is small and deterministic.

### 2. Keep multiplier set shift-add friendly

The subgroup ratios are ideal for hardware:

- `1.00 = P`
- `1.25 = P + P/4`
- `1.50 = P + P/2`
- `1.75 = P + P/2 + P/4`

So the decode side can use shift-add rather than a general multiplier.

### 3. Calibrate thresholds offline, not online

The threshold `T` and optional correction LUT should be learned once from representative data.

A practical flow is:

1. Run the original exact-search `nvesm2` offline.
2. Collect tuples `(a1, a2, s_base, k_exact)`.
3. Fit a tiny LUT or piecewise-threshold rule that predicts `k_exact`.
4. Freeze that rule into RTL.

This gives you most of the benefit of search without carrying search into the hardware path.

### 4. Separate weight and activation policies

Literature and practical systems both suggest the best online format for activations may differ from the best offline format for weights.

My recommendation:

- weights: exact `Sg-EM-2bit` offline search is fine
- activations: use the LUT estimator, or even switch to top-1 metadata if you can afford format changes

## If format change is allowed

If you are **not** forced to keep exact `nvesm2` metadata semantics, then a stronger hardware-oriented option is:

- keep subgroup/block shared scale for weights
- replace activation-side subgroup scale search with **top-1 extra mantissa metadata** per subgroup

This is close to the design direction reported by recent work such as M2XFP:

- static weights benefit from subgroup-level 2-bit mantissa refinement with adaptive scale
- dynamic activations benefit more from very simple online encoding with minimal selection logic

So if your target is a real accelerator rather than a drop-in software emulation of `nvesm2`, I would seriously consider:

- weights: `Sg-EM-2bit` (very close to original nvesm2 spirit)
- activations: top-1 element refinement instead of subgroup ratio search

## Recommended decision

If you want the **lowest-risk drop-in hardware-friendly replacement** for `nvesm2`, use this:

1. Keep the existing `nvesm2` storage format.
2. Keep exact search only for offline weights.
3. For any online quantization path, replace 4-way ratio search with:
   - subgroup `amax` / optional `top2`
   - direct 2-bit ratio quantization
   - one outlier-aware correction LUT

That is the design I would prioritize first.

If you want the **best long-term accelerator-oriented design**, use this:

1. Weights: subgroup 2-bit mantissa refinement with offline adaptive search.
2. Activations: switch to top-1 metadata encoding rather than subgroup search.

## Minimal pseudo-code for the proposed online encoder

```python
# subgroup size = 8
# ratios = [1.0, 1.25, 1.5, 1.75]

a = abs(x_subgroup)
a1, idx1 = top1(a)
a2 = top2(a)

# s_base is the already-quantized block scale
r_raw = a1 * recip_6sbase  # recip_6sbase = 1 / (6 * s_base)
k0 = clamp(round(4 * (r_raw - 1.0)), 0, 3)

outlier_flag = (a1 > T * a2)
k = lut[k0][outlier_flag]
ratio = 1.0 + 0.25 * k

s_final = s_base * ratio
q = cast_to_fp4(x_subgroup / s_final)
```

## Reference links

### Standards / official docs

- OCP MX specification: https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf
  - Defines MX block scaling, FP4 E2M1 grid, and required rounding behavior.
- NVIDIA Transformer Engine NVFP4 docs: https://nvidia.github.io/TransformerEngine/features/low_precision_training/nvfp4/nvfp4.html
  - Useful for Blackwell-style hierarchical scaling, block size 16, and practical layout constraints.
- NVIDIA TensorRT-LLM quantization docs: https://nvidia.github.io/TensorRT-LLM/torch/features/quantization.html
  - Practical deployment-facing documentation mentioning FP8/NVFP4 support.

### Papers

- Microscaling Data Formats for Deep Learning (arXiv 2310.10537): https://arxiv.org/abs/2310.10537
  - Foundational MX paper.
- AMXFP4: Taming Activation Outliers with Asymmetric Microscaling Floating-Point for 4-bit LLM Inference (arXiv 2411.09909): https://arxiv.org/abs/2411.09909
  - Important for understanding activation outliers and why symmetric subgroup scaling can fail.
- OPAL: Outlier-Preserved Microscaling Quantization Accelerator for Generative Large Language Models (arXiv 2409.05902): https://arxiv.org/abs/2409.05902
  - Hardware-software co-design focused on preserving outliers efficiently.
- Oscillation-Reduced MXFP4 Training for Vision Transformers (arXiv 2502.20853 / ICML 2025): https://arxiv.org/abs/2502.20853
  - More training-oriented, but useful for understanding stability issues in FP4 microscaling.
- M2XFP: A Metadata-Augmented Microscaling Data Format for Efficient Low-bit Quantization (arXiv 2601.19213): https://arxiv.org/abs/2601.19213
  - Most relevant recent direction for metadata-augmented microscaling and lightweight hardware support.

### Repositories

- Microsoft microxcaling: https://github.com/microsoft/microxcaling
  - MX emulation library and integration guide.
- NVIDIA Model Optimizer: https://github.com/NVIDIA/Model-Optimizer
  - Practical NVFP4 quantization framework and export flow.
- aiha-lab/MX-QLLM: https://github.com/aiha-lab/MX-QLLM
  - Open implementation for MXFP4 / AMXFP4 style LLM inference experiments.
- thu-ml/TetraJet-MXFP4Training: https://github.com/thu-ml/TetraJet-MXFP4Training
  - Open implementation for MXFP4 training stabilization.
- SJTU-ReArch-Group/M2XFP_ASPLOS26: https://github.com/SJTU-ReArch-Group/M2XFP_ASPLOS26
  - Repository referenced by the M2XFP paper.

## Bottom line

For `nvesm2`, the right hardware optimization is not a different heuristic score over 4 candidates. The right optimization is to eliminate multi-candidate search entirely and replace it with a direct subgroup-statistic estimator plus a tiny correction LUT.
