# Good/Bad L1 Group 度量分析

## 0. 问题概述

对 L1 group（16 元素共享一个 FP4 scale）按 MSE 的 10th/90th percentile 分为 good/bad，
将每个 L1 group 拆成两半（各 8 元素），比较两半之间的多种度量。

**Gaussian 数据观测结果（good group 相对 bad group）：**

| 度量 | good 方向 | 直觉预期 | 是否符合 |
|------|----------|---------|---------|
| position-aligned L1（标为 wasserstein） | 较小 | 较小 | ✓ |
| mutual information | 较小 | 较大 | ✗ |
| kl_elem（elem/sum） | 较大 | 较小 | ✗ |
| kl_hist（histogram） | 较小 | 较小 | ✓ |
| kurtosis | 较大 | 较小 | ✗ |

**真实数据观测结果：所有方向恰好与 Gaussian 相反。**

---

## 1. 代码级问题：wasserstein_empirical 不是 Wasserstein

```python
def wasserstein_empirical(p_vals, q_vals):
    p_sorted, _ = torch.sort(p_vals, dim=1)
    q_sorted, _ = torch.sort(q_vals, dim=1)
    p_sorted = p_vals    # ← 覆盖了排序结果
    q_sorted = q_vals    # ← 覆盖了排序结果
    return torch.mean(torch.abs(p_sorted - q_sorted), dim=1).mean().item()
```

排序结果被原始值覆盖。当前实际计算的是 **逐位置配对的平均 L1 距离**，
对"哪个位置有大值"敏感；而真正的 1D empirical Wasserstein（先排序再配对）只比较分布形状。

---

## 2. 核心框架："联系紧密" 需要拆成两个正交维度

"好 group 的两半应该联系更紧密" 这个推断隐含假设所有度量在测同一件事。
但实际上这些度量分布在两个独立的维度上：

| 度量 | 测什么 | 对位置敏感 | 对分布形状敏感 |
|------|--------|----------|-------------|
| position-aligned L1 | 逐位置绝对差 | **是** | 否 |
| mutual information | 逐位置统计依赖 | **是** | 否 |
| kl_elem | 位置级"质量分配模式" | **是** | 间接 |
| kl_hist | 边缘分布形状 | **否** | **是** |
| kurtosis | 尾部轻重 | N/A | **是** |

**关键洞察**：两半可以拥有极其相似的边缘分布形状（kl_hist 小），
但大值落在不同位置（kl_elem 大），且逐位置没有统计依赖（MI 小）。
这三件事可以同时为真，彼此之间不矛盾。

---

## 3. Gaussian 数据：什么决定了哪些 L1 group 是 "好" 的？

### 3.1 MSE 的分解与 "混合效应"

FP4 E2M1 的量化 MSE 可以分解为：

    MSE = s² × (1/16) × Σ(rounding_error_i)²

其中 s = max|x_i| / 6 是 scale。

FP4 E2M1 的格点结构（正半轴）：0, 0.5, 1, 1.5, 2, 3, 4, 6
- 0-2 区间：gap = 0.5（fine）
- 2-4 区间：gap = 1（medium）
- 4-6 区间：gap = 2（coarse）

两个竞争因素：
- s² 正比于 max²（偏好 max 小的 group → scale 小 → MSE 小）
- 当 max 大时 normalized values 被压缩到 0 附近 → 落在 fine 区域 → rounding error 小

它们**部分抵消**。具体计算（16 个 |N(0,1)| 样本的 max 分布）：

| max / σ | MSE 近似值 (× σ²) | 备注 |
|---------|-------------------|------|
| 1.50 (10th pctl) | 0.0057 | scale 效应占优 → MSE 低 |
| 2.15 (median) | 0.0096 | |
| 2.72 (90th pctl) | 0.0140 | scale 效应占优 → MSE 高 |

scale 效应占主导但不是压倒性的 → MSE 筛选不等价于 max 筛选。

**这意味着 "好 group" 是两类 group 的混合体：**

- **Mode A（主体）**：max 较小 → normalized 后值分布宽（σ_norm ≈ 4）→ 更均匀，无极端 outlier
- **Mode B（少数但影响大）**：max 不小，但值恰好落在 FP4 格点附近 →
  低 rounding error 弥补了高 scale → 也被选为 good → 保留了 outlier 结构

**这个混合效应是理解 kl_elem 和 kurtosis 反直觉结果的关键。**

### 3.2 Normalized 后的分布特征

对于 good group（max ≈ 1.5σ）：
- s = max/6 → 除以 s 后 σ_norm = 6σ/max = 6/1.5 = 4
- 值从 truncN(0, 16, -6, 6)，即 Gaussian 在 ±1.5σ 处截断
- 分布较宽，近似均匀

对于 bad group（max ≈ 2.7σ）：
- σ_norm = 6/2.7 ≈ 2.2
- 值从 truncN(0, 4.84, -6, 6)，截断在 ±2.7σ（几乎不截断）
- 分布集中在 [-4, 4]，max 处有一个 outlier at ±6

---

## 4. 逐度量解析（Gaussian 数据）

### 4.1 Position-aligned L1 较小 ✓

- Good groups（Mode A 主导）：两半都没有极端 outlier →
  逐位置差值没有被单个极端 pair 主导 → 平均 L1 较小
- Bad groups：outlier 只在其中一半 → 对应位置差值很大 → 平均 L1 较大

这个结果方向正确，但需注意实际算的不是 Wasserstein。

### 4.2 MI 较小 ✓（不违反直觉，需要修正直觉）

**为什么直觉"联系紧密 → MI 大"在这里不成立：**

对于 i.i.d. Gaussian，L1 group 的两半本身**完全独立**。
但 conditioning on MSE 会引入间接依赖：

- **Bad groups（max 极端）**：max 只能在其中一半 →
  "如果第一半包含 max，第二半大概率不包含极端值" →
  两半之间存在 **selection-induced anti-correlation** →
  MI 不区分正负相关，只捕捉依赖强度 → **MI 偏大**

- **Good groups（max 不极端）**：conditioning 效应弱 →
  两半几乎保持独立 → **MI 偏小**

**结论**："联系紧密 → MI 大" 只在存在**真正的结构性位置耦合**时成立（如真实数据中的通道共激活），
不在 "仅同分布的 i.i.d. 抽样" 时成立。好的 Gaussian group 恰恰因为没有极端 max，
selection effect 弱，两半更接近真正独立 → MI 自然更小。

### 4.3 kl_hist 较小 ✓

kl_hist 比较两半的边缘分布（histogram 形状，与位置无关）：

- **Good groups（Mode A）**：σ_norm 大，两半都从宽分布中抽样 →
  histogram 形状相似 → kl_hist 小
- **Bad groups**：一半包含 outlier（histogram 有远端 peak），另一半没有 →
  histogram 形状不同 → kl_hist 大

### 4.4 kl_elem 较大 ✗ — 混合效应的体现

`elem_prob(x)` = |x_i| / Σ|x_j|，kl_elem 测的是两半在 **相同位置** 上的
"质量分配模式"是否一致。

**对纯 Mode A（max 小，spread 宽）**：
- 两半的 elem_prob 都接近 uniform (1/8, 1/8, ...) → KL 应该很小

**但 Mode B 的 good groups（max 不小，grid alignment 好）** 依然有 outlier 结构：
一半有大值，一半没有 → kl_elem 偏大。Mode B 虽然数量少，但对 KL 的贡献
被放大（KL 对极端概率比值非常敏感）→ **平均 kl_elem 被 Mode B 拉高**。

同时，**bad groups 也是混合体**：
- 一部分是 max 极大的（outlier structure → 高 kl_elem）
- 一部分是 max 正常但 grid alignment 差的（没有 outlier → 较低 kl_elem）

如果 bad groups 中 "grid alignment 差但无 outlier" 的比例较高，
bad 的 kl_elem 反而被这些 group 拉低 → 形成 good > bad 的结果。

**验证方法**：把 good groups 按 max 分成 "small-max" 和 "large-max" 两个子集，
分别看 kl_elem。如果混合效应正确，large-max 子集的 kl_elem 应显著更高。

### 4.5 Kurtosis 较大 ✗ — 混合效应 + 小样本偏差

理论上各 mode 的 population excess kurtosis：

| 类型 | 分布特征 | 近似 excess kurtosis |
|------|---------|---------------------|
| Good Mode A（σ_norm=4, 截断 ±1.5σ） | 接近均匀 | ≈ -0.3 到 -0.6 |
| Good Mode B（有 outlier 但 MSE 低） | 有极端值 | 正值 |
| Bad 半含 outlier | 一个 2.7σ 的 outlier | ≈ 0 |
| Bad 半无 outlier | 近 Gaussian | ≈ 0 |

Mode A 和 Mode B 混合后的平均 kurtosis 取决于两者的占比。
如果 Mode B 的正 kurtosis 足够大且占比不太低，就能把 good 的平均 kurtosis 拉到比 bad 更高。

另外，n=8 的 sample excess kurtosis 有显著负偏：
E[Ĝ₂] ≈ κ - 6/(n+1) = κ - 0.67（对 Gaussian 而言），
且偏差大小依赖于真实分布。对截断 normal 和含 outlier 的 normal，偏差不同。
在 n=8 这么小的样本上，偏差效应足以影响 good/bad 之间微弱的理论差距。

**注意**：n=8 的 sample kurtosis std error ≈ 1.7（for Gaussian），
good 和 bad 之间的差异可能很小，方向依赖于实验具体参数。

---

## 5. 为什么真实数据结果完全相反？

### 5.1 根本原因：Gaussian 和真实数据中 "好量化" 的成因完全不同

**Gaussian 数据中的 "好 group"：**
- 本质是 **随机运气** — max 碰巧小，或值碰巧对齐 FP4 格点
- 两半之间没有结构性联系（i.i.d.）
- 条件依赖弱 → MI 小

**真实数据中的 "好 group"：**
真实激活有强烈的通道结构：
1. **稀疏性**：大量 near-zero 的激活（post-ReLU 或 gate 导致）
2. **通道异方差**：不同 channel 的 magnitude 差异巨大
3. **空间相关性**：相邻 channel 的激活可能高度相关

真实数据中低 MSE 的 L1 group 最可能是以下两类之一：

**(a) Near-zero group（主导）**：16 个 channel 全部接近 0
- scale 极小 → MSE 极小（trivially well-quantized）
- normalization 后微小噪声被放大 → 两个噪声 pattern 无关
- kl_hist 大（噪声 histogram 形状随机不同）
- kl_elem 小（所有值都差不多）
- kurtosis 低（values 集中在 0 附近，无重尾）

**(b) 结构化匹配 group**：16 个 channel 的 magnitude 恰好都相似
- shared scale 效果好 → MSE 低
- 两半有相似的激活 pattern（相邻 channel 共激活）→ MI 大
- kl_elem 小（位置上的质量分配模式相似）

### 5.2 真实数据中的 "坏 group"

- 高动态范围：一些 channel 极其活跃，另一些接近 0
- shared scale 被活跃 channel 主导 → near-zero channel 量化误差大
- 两半的激活 pattern 可能不同（一半活跃一半沉默）
- kl_elem 大（位置级差异大）
- kurtosis 高（有极端 outlier channel）
- MI 低（没有位置级结构耦合）

### 5.3 反转汇总

| 度量 | Gaussian good | 真实 good | 反转的原因 |
|------|-------------|----------|-----------|
| MI | 小（两半独立，selection effect 弱） | 大（相邻 channel 共激活） | i.i.d. vs 结构耦合 |
| kl_elem | 大（Mode B 混合效应） | 小（通道 pattern 匹配） | 随机 vs 结构化相似 |
| kl_hist | 小（同分布抽样） | 大（near-zero 组的噪声放大） | 好组本质不同 |
| kurtosis | 大（Mode B 混合效应） | 小（near-zero → 无重尾） | 好组的本质不同 |
| position-aligned L1 | 小（无 extreme outlier 差） | 大（两半都活跃，差异绝对值大） | "好"的含义不同 |

**核心结论**：Gaussian 的 "好" = 随机运气（小 max + 碰巧 grid alignment），
真实数据的 "好" = 结构性匹配（通道相似 or near-zero）。
两种 "好" 在统计特征上几乎处处相反。

---

## 6. 原始直觉的修正

### 正确的部分

- "kl_hist 小 → 边缘分布更像" — 在 Gaussian 上完全成立
- "Position-aligned L1 小 → 两半逐位置更接近" — 方向碰巧正确（但实现有 bug）

### 需要修正的部分

1. **MI 不测 "分布相似性"，只测 "位置级依赖"。**
   两半同分布但独立 → MI=0。
   "联系紧密 → MI 大" 只在存在真正的位置耦合时成立。

2. **kl_elem 和 kl_hist 测的是正交维度。**
   kl_hist 比较 "what values exist"（分布形状），
   kl_elem 比较 "which positions have what values"（质量分配模式）。
   两者完全可以方向相反。

3. **n=8 的 sample kurtosis 非常不稳定。**
   MSE 筛选引入混合效应（两类不同的 "好 group"），
   平均 kurtosis 的方向取决于两类 mode 的占比和小样本偏差。

---

## 7. 建议的验证实验

### 验证 1：拆分 good groups 验证混合效应

把 good groups 按 L1 group 的 max 值分成 "small-max" 和 "large-max" 两个子集，
分别计算 kl_elem 和 kurtosis。

- 如果混合效应正确："large-max" 子集的 kl_elem 和 kurtosis 应显著高于 "small-max" 子集
- 这能直接验证 Mode A / Mode B 的假设

### 验证 2：修正 Wasserstein 实现

去掉 `wasserstein_empirical` 中覆盖排序结果的两行。
同时保留当前指标，改名为 `paired_l1_distance`。
对比修正前后的结果趋势是否一致。

### 验证 3：MI 的 permutation 对照

对 `q_vals` 沿 dim=1 随机 shuffle 后重算 MI：
- 如果 shuffle 后 MI 几乎不变 → 当前 MI 主要在测分布相似性
- 如果 shuffle 后 MI 大幅下降 → 当前 MI 主要在测位置耦合
- 对 Gaussian 数据，预期 shuffle 影响小（因为本来就是 i.i.d.）
- 对真实数据，预期 shuffle 影响大（因为有通道结构）

### 验证 4：kl_elem vs kl_elem_sorted 对比

代码中已有 `kl_elem_sorted`（先对绝对值排序再算 elem_prob 再算 KL）：
- 若 kl_elem 大但 kl_elem_sorted 小 → 确认是 "分布相似但位置不匹配"
- 若两者同向 → 位置不是主要因素

### 验证 5：真实数据的 channel-shuffle 对照

把每个 L1 group 的 16 个值随机打乱后重做所有实验：
- 如果打乱后结果趋势变得接近 Gaussian → 确认 "通道结构" 是反转的原因
- 如果打乱后无变化 → 反转来自分布特征而非位置结构

### 验证 6：统计 good/bad group 的 structural descriptors

在两类数据上分别统计每个 good/bad L1 group 的：
- max / std 比（动态范围指标）
- 稀疏度（|x_i| < threshold 的比例）
- 两半的 Pearson 相关系数
- 两半的 variance ratio

目标是刻画 "低 MSE group" 到底偏向 "均匀平滑" 还是 "结构化可预测"。
