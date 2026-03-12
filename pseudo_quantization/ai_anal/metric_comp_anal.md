  ## Summary

  当前结果很可能不是单一“联系紧密/不紧密”能解释的，而是 3 类因素叠加：

  - 指标语义不一致：你现在的 MI 测的是“按位置配对后的依赖性”，两种 KL 分别更接近“质量分配相似性”和“边缘分布形状相似性”，不是同一个概念。
  - 实现上有一个关键偏差：wasserstein_empirical() 里排序结果被覆盖了，当前实际算的是按位置对齐的平均 L1，不是 Wasserstein distance。
  - 数据机制不同：Gaussian 数据近似 i.i.d.，真实激活数据有强异方差、长尾、稀疏、符号结构和通道语义，导致“好量化 group”的成因不同，所以指标方向完
    全可能反过来。

  ## Key Changes

  - 先把当前实验的可解释性问题锁定为 4 个具体假设。
  - 假设 1：wasserstein 排序异常主要来自实现问题。
      - 代码位置：pseudo_quantization/mxq/quantize/kl_kur_comp_with_gaussian.py 和 pseudo_quantization/mxq/quantize/kl_kur_comp_with_realinput.py
      - 现状：torch.sort(...) 后立刻执行 p_sorted = p_vals、q_sorted = q_vals，所以现在不是 Wasserstein。
      - 预期：修正后，若“好 group 的两半更像同一分布”，真实 Wasserstein 应更稳定地偏小。
  - 假设 2：MI 小并不表示“两半不相似”，只表示“逐元素位置上的统计依赖弱”。
      - 现状：mutual_information_hist(p_vals, q_vals) 把第 i 个位置的 p[i] 和 q[i] 当成联合样本。
      - 含义：如果两半只是“分布相似”但元素顺序/位置无关，MI 可以很小，同时 histogram KL 仍然很小。
      - 因此你原本“联系更紧密 => MI 更大”的推断只在“存在位置级耦合/函数关系”时成立，不在“仅同分布”时成立。
  - 假设 3：kl_elem 和 kl_hist 本来就可能相反。
      - kl_elem：先取 abs，再按行归一化求概率，测的是组内 8 个位置的“质量分配模式”是否一致；对哪几个位置大、哪几个位置小很敏感。
      - kl_hist：先对每对 group 做联合 min-max 归一化，再分桶；测的是两半的数值直方图是否相似；对位置置换基本不敏感。
      - 所以一种很自然的情形是：
          - 两半拥有相似边缘分布形状 -> kl_hist 小
          - 但大值落在不同位置 -> kl_elem 大
  - 假设 4：Gaussian 与真实数据方向相反，来自“好量化”的定义在两类数据里不一样。
      - Gaussian 下，低 MSE 的 l1_group 更可能是“整体动态范围均匀、两半都不极端、分布更像”，因此 histogram KL 小、位置耦合弱、MI 小。
      - 真实激活下，低 MSE 的 l1_group 更可能是“共享同一个 scale 仍能被好量化的结构化 pattern”，例如一半主导 scale、另一半跟随或稀疏，这会让位置
        级依赖更强，MI 变大，但边缘分布未必更像。
      - 真实数据还有通道语义和时序/token 顺序，group 两半不是可交换的 i.i.d. 样本，这和 Gaussian 本质不同。

  ## Test Plan

  - 验证 1：修正 Wasserstein 实现后，重新比较 good/bad。
      - 计算真正的 1D empirical Wasserstein：先排序再做配对差的平均。
      - 同时保留当前指标，改名为 paired_l1_distance，避免混淆。
  - 验证 2：把“分布相似”和“位置耦合”拆开测。
      - 保留现有 kl_hist 作为边缘分布相似性。
      - 对 MI 增加一个对照：对 q_vals 随机打乱位置后再算 MI。
      - 如果打乱后 MI 变化很大，说明当前 MI 主要在测位置耦合，不是分布接近。
  - 验证 3：检查位置因素是否导致 kl_elem 偏大。
      - 现有脚本已经有 kl_elem_sorted，它更接近“忽略位置，只比较幅值排序后的形状”。
      - 若 kl_elem 大但 kl_elem_sorted 小，说明问题主要是位置错配，不是幅值形状不相似。
  - 验证 4：解释 Gaussian/realinput 反转。
      - 在两类数据上分别统计每个 good/bad group 的：
          - 方差/最大值比
          - 稀疏度（接近 0 的比例）
          - 符号一致性
          - 两半的相关系数
          - kurtosis
      - 目标是看“低 MSE group”到底偏向“均匀平滑”还是“结构化可预测”。
  - 验证 5：做最小反例实验，确认每个指标各自偏好什么。
      - 反例 A：两半是同一组数值的随机置换。
          - 预期：kl_hist 小，真实 Wasserstein 小，kl_elem 大，MI 小。
      - 反例 B：q = a * p + b 单调关系强但边缘分布不同。
          - 预期：MI 大，但 histogram/elem KL 不一定小。
      - 反例 C：两半都稀疏，且大值总落在同一位置。
          - 预期：MI 可能大，kl_elem 可能小，但 histogram KL 取决于数值范围。
      - 这 3 个反例足以说明“联系紧密”不是单一轴。

  ## Assumptions

  - 默认把“联系更紧密”拆成两个概念：
      - 分布更接近
      - 位置/元素更相关
  - 默认认为你当前最该修的不是量化逻辑，而是指标解释和 wasserstein 实现。
  - 默认认为 Gaussian 与真实数据出现反转是合理现象，不优先按“实验错了”处理；但 wasserstein 命名/实现目前确实有问题。
  - 不做接口级修改；如果进入实现阶段，只需要在上述两个分析脚本里修正指标实现和补充对照指标
