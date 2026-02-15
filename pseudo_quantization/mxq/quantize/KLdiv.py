import numpy as np

def manual_kl_divergence(p_data, q_data):
    # 1. 转换为 Numpy 数组并平坦化
    p = np.asarray(p_data, dtype=np.float64)
    q = np.asarray(q_data, dtype=np.float64)

    # 2. 归一化为概率分布 (Sum to 1)
    p = p / np.sum(p)
    q = q / np.sum(q)

    # 3. 加上 epsilon 避免 log(0) 或除以 0
    eps = 1e-10
    p = np.clip(p, eps, 1)
    q = np.clip(q, eps, 1)
    print(p, q)

    # 4. 根据公式计算
    return np.sum(p * np.log(p / q))

# 测试
# P = [0.1, 0.9, 0.0]
# Q = [0.2, 0.7, 0.1]
P = [1,1,1,1,1]
Q = [1,1,1,1,1]
print(f"KL Divergence: {manual_kl_divergence(P, Q):.6f}")


