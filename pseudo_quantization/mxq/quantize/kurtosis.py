import numpy as np
from scipy.stats import kurtosis
import matplotlib.pyplot as plt

# 设置随机种子保证结果可复现
np.random.seed(42)

# 1. 生成两个方差不同的正态分布
# 标准正态分布 (均值0, 标准差1, 方差1)
dist_1 = np.random.normal(0, 1, 100000)
# 方差更小的正态分布 (均值0, 标准差0.5, 方差0.25)
dist_2 = np.random.normal(0, 0.5, 100000)

# 2. 计算超额峰度 (Excess Kurtosis)
# fisher=True 返回的是 (常规峰度 - 3)，正态分布理论值应为 0
k1 = kurtosis(dist_1, fisher=True)
k2 = kurtosis(dist_2, fisher=True)

print(f"标准正态分布 (std=1.0) 的超额峰度: {k1:.4f}")
print(f"小方差正态分布 (std=0.5) 的超额峰度: {k2:.4f}")

# 3. 绘图比较
plt.figure(figsize=(10, 6))
plt.hist(dist_1, bins=100, density=True, alpha=0.5, label='Std=1.0', color='blue')
plt.hist(dist_2, bins=100, density=True, alpha=0.5, label='Std=0.5', color='red')
plt.title("Comparison of Normal Distributions with Different Variances")
plt.xlabel("Value")
plt.ylabel("Density")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
plt.savefig("kurtosis.png", dpi=150)
