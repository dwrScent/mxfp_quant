# nvesm2 Search-Preserving Hardware-Friendly Notes

## Scope

This note focuses on **hardware-friendly optimization that preserves the core of the original nvesm2 algorithm**:

- scale candidates still exist
- candidates are still compared by an error metric
- the final scale is still selected by a search-select process

The goal is **not** to replace search with a pure predictor, but to reduce the search cost.

## Local code anchor

- Original search-based implementation: `quant_func.py` commented `get_quant_nvesm2`
- Current active implementation: `quant_func.py` active `get_quant_nvesm2`

The original version evaluates all 4 ratio candidates:

- `r in {1.0, 1.25, 1.5, 1.75}`
- dequantizes each candidate
- computes an error metric
- picks argmin

That is the right accuracy-oriented baseline.

## What I learned from recent papers

### 1. Shared-scale FP4 formats are attractive, but outliers and asymmetry are the main failure modes

- MX paper (2023) establishes the shared-scale microscaling direction.
- AMXFP4 (2024/2025) shows that plain shared-scale MXFP4 can lose accuracy because suppressing outliers increases group-wise asymmetry.
- OPAL (2024) shows that preserving a few outliers explicitly is effective in hardware-software co-design.

Practical implication for nvesm2:

- candidate preselection should be **outlier-aware**
- the best candidate is usually near the scale suggested by the subgroup extreme statistics, but strong outliers can bias that estimate upward

### 2. Recent hardware-oriented work does not try to keep expensive online optimization intact

- M2XFP (2026) explicitly pushes toward **simple online encoding** with a lightweight metadata unit.

Inference from that direction:

- if we must keep nvesm2's search-select nature, the hardware path should at least shrink from `4-way full evaluation` to `top-k reduced evaluation`
- a small metadata/predictor stage before reduced search is consistent with current hardware-oriented microscaling research

### 3. Hardware-friendly design should reduce candidate count first, arithmetic precision second

OPAL and M2XFP both suggest the right system-level strategy:

- simplify online selection logic
- isolate exceptional cases such as outliers
- keep the datapath regular for the common case

For your nvesm2 setting, that means:

- first reduce 4 candidates to 2
- then approximate MSE for those 2
- avoid carrying 4 full normalize-quantize-error pipelines

## Recommended search-preserving flow

I recommend this specific pipeline.

### Stage 0. Base scale generation

This stays aligned with your original implementation.

1. For each group, compute base scale:
   - `s_base = Q_fp8(amax(group) / 6)`
2. Split the group into subgroups of 8.

### Stage 1. Cheap feature extraction per subgroup

For subgroup `x = {x_0, ..., x_7}` compute:

- `a1 = max_i |x_i|`
- `a2 = second_max_i |x_i|`
- `rho = a2 / a1`
- `u = a1 / (6 * s_base)`

Optional cheap counters:

- `n_hi(t) = count(|x_i| > t)`
- use `t = 6 * s_base * r_center`

These are cheap reductions / comparisons and can be reused by later control logic.

### Stage 2. Candidate preselection (4 -> 2)

Let candidate code `k in {0,1,2,3}` correspond to:

- `r(k) = 1.0 + 0.25 * k`

#### Step 2.1 Center candidate

Use the subgroup maximum to generate a center candidate:

- `k_center = clamp(round(4 * (u - 1.0)), 0, 3)`

This means: if `a1` is close to `6*s_base*1.25`, the center candidate becomes `1.25`, etc.

#### Step 2.2 Neighbor direction

Use outlier sensitivity to decide whether the second candidate should be above or below the center.

Rule:

- if `rho < tau_outlier`, choose downward neighbor
- else if `n_hi(6*s_base*r(k_center)) > 0`, choose upward neighbor
- else choose downward neighbor

Suggested initial threshold:

- `tau_outlier = 0.5 ~ 0.7`

So candidate set `C` becomes:

- strong outlier case: `C = {k_center, max(k_center-1, 0)}`
- likely under-scaled case: `C = {k_center, min(k_center+1, 3)}`
- otherwise: `C = {k_center, max(k_center-1, 0)}`

This keeps the true search-select structure, but reduces full evaluation from 4 candidates to 2.

## Why this preselection is reasonable

The 4 candidate scales are ordered and very close:

- `1.00`
- `1.25`
- `1.50`
- `1.75`

So the optimal candidate is usually either:

- the candidate nearest to the normalized subgroup max, or
- one adjacent neighbor when the subgroup has a single large outlier or mild under-scaling

This is exactly the structure exploited by the preselector.

## Stage 3. Approximate-MSE comparison for the 2 candidates

For each candidate `k in C`, define:

- `r = r(k)`
- `s = s_base * r`
- `z_i = |x_i| / s`
- `q_i = Q_fp4(z_i)`
- true normalized-domain error: `e_i = z_i - q_i`

The true candidate objective is:

- `MSE_true(k) = s^2 * sum_i e_i^2`

Because `s_base^2` is common across candidates, candidate comparison only needs:

- `Score_true(k) ∝ r(k)^2 * sum_i e_i^2`

### Hardware-friendly approximate-MSE

Instead of full-precision `e_i^2`, use a small LUT over bucketed normalized error.

#### Step 3.1 Error bucketization

For each element:

- `d_i = |z_i - q_i|`

Bucket `d_i` into 4 levels:

- `B0: d_i < 1/8`
- `B1: 1/8 <= d_i < 1/4`
- `B2: 1/4 <= d_i < 1/2`
- `B3: d_i >= 1/2`

#### Step 3.2 Squared-error surrogate LUT

Map buckets to an approximate squared cost:

- `phi(B0) = 0`
- `phi(B1) = 1`
- `phi(B2) = 4`
- `phi(B3) = 16`

This is intentionally shaped like `err^2`.

A slightly finer 8-bin version is also possible, but 4 bins are enough as a first RTL target.

#### Step 3.3 Candidate score

Use integer weights for `r^2`:

- `r^2 * 16 = {16, 25, 36, 49}` for `r = {1.0, 1.25, 1.5, 1.75}`

Then compute:

- `Score(k) = ratio_sq_lut[k] * sum_i phi(bucket(d_i)))`

Optional overflow penalty:

- `Score(k) += lambda_sat * sat_count(k)`

where:

- `sat_count(k) = count(z_i > 6.0)`

This overflow penalty is useful because a candidate with subgroup overflow can look deceptively acceptable under a coarse error surrogate.

### Final select

- `k_best = argmin_{k in C} Score(k)`
- output subgroup scale `s_base * r(k_best)`

## Why this approximate MSE is much closer to the original algorithm

Compared with replacing search by a direct predictor, this version keeps the original core:

- candidates are explicitly generated
- each candidate gets an error score
- the selected candidate is still the one minimizing an approximate error

What changed is only:

- `4 candidates -> 2 candidates`
- `full MSE -> LUT-approximated MSE`

So this is a compression of the search engine, not a replacement of the search engine.

## Concrete end-to-end flow

### Input

- subgroup of 8 values
- base scale `s_base`

### Control path

1. Compute `a1`, `a2`, `rho`, `u`
2. Compute `k_center`
3. Choose neighbor direction from `rho` and optional `n_hi`
4. Build candidate set `C` with 2 candidates

### Datapath for each candidate

1. Read reciprocal from LUT for `1 / (s_base * r)`
2. Normalize 8 values
3. Quantize each normalized value to nearest FP4 grid point
4. Compute `d_i = |z_i - q_i|`
5. Bucketize `d_i`
6. Accumulate surrogate cost
7. Multiply by `ratio_sq_lut[k]`
8. Add overflow penalty if needed

### Final step

- compare the two candidate scores
- emit `k_best`

## Further hardware reductions

If the 2-candidate path is still too expensive, reduce cost in this order.

### Option A. Sequential 2-candidate reuse

Do not instantiate 2 parallel lanes.

- evaluate candidate 1
- store partial score
- evaluate candidate 2 in the same lane
- compare at the end

This doubles latency of the selection stage, but greatly reduces area.

### Option B. Shared comparator tree for FP4 quantization

Because the FP4 grid is fixed, use a shared boundary-comparison tree rather than general-purpose arithmetic.

### Option C. Replace exact nearest-FP4 with interval code lookup

The FP4 boundaries are fixed:

- `0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0`

So quantization is naturally comparator-based.

### Option D. Teacher-student calibration for better preselection

Offline, run the original full-search algorithm and collect:

- `a1`
- `a2`
- `rho`
- `k_opt`

Then tune:

- `tau_outlier`
- `lambda_sat`
- bucket thresholds
- bucket weights

to maximize `top-2 hit rate` and match original `k_opt`.

This is the most practical way to preserve accuracy.

## My recommended first implementation

If I were implementing this in hardware first, I would use:

1. `k_center` from subgroup `amax`
2. second candidate from `rho = a2/a1`
3. 2-candidate reduced search
4. 4-bin approximate-MSE LUT
5. overflow penalty

That is the smallest design that still preserves the original nvesm2 philosophy.

## Learning links

### arXiv papers

- Microscaling Data Formats for Deep Learning
  - https://arxiv.org/abs/2310.10537
- OPAL: Outlier-Preserved Microscaling Quantization Accelerator for Generative Large Language Models
  - https://arxiv.org/abs/2409.05902
- AMXFP4: Taming Activation Outliers with Asymmetric Microscaling Floating-Point for 4-bit LLM Inference
  - https://arxiv.org/abs/2411.09909
- M2XFP: A Metadata-Augmented Microscaling Data Format for Efficient Low-bit Quantization
  - https://arxiv.org/abs/2601.19213

### Repositories

- Microsoft microxcaling
  - https://github.com/microsoft/microxcaling
- MX-QLLM
  - https://github.com/aiha-lab/MX-QLLM
- M2XFP repository
  - https://github.com/SJTU-ReArch-Group/M2XFP_ASPLOS26

### Google Scholar query links

- Google Scholar query: microscaling fp4 hardware quantization
  - https://scholar.google.com/scholar?q=microscaling+fp4+hardware+quantization
- Google Scholar query: approximate mse quantization hardware accelerator
  - https://scholar.google.com/scholar?q=approximate+MSE+quantization+hardware+accelerator
- Google Scholar query: activation outlier microscaling fp4
  - https://scholar.google.com/scholar?q=activation+outlier+microscaling+fp4
- Google Scholar query: metadata augmented microscaling low-bit quantization
  - https://scholar.google.com/scholar?q=metadata+augmented+microscaling+low-bit+quantization

## Bottom line

A hardware-friendly nvesm2 should not delete the search-select step. It should compress it:

- preselect 2 candidates from 4 using outlier-aware subgroup statistics
- compare those 2 candidates with a hardware-friendly approximate-MSE
- keep argmin selection

That is the closest hardware-friendly version of the original nvesm2 idea.
