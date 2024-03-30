import torch
import torch.nn as nn
import kmeans_parallel


batch_size = 128
N = 128
K = 16

def random_initializer(input_data, n_clusters):
    batch_size, N = input_data.shape
    centroids = torch.empty(batch_size, n_clusters, device=input_data.device, dtype=input_data.dtype)

    for batch_idx in range(batch_size):
        # 随机打乱输入数据的索引
        shuffled_indices = torch.randperm(N, device=input_data.device)
        
        # 从打乱的索引中选择前K个作为初始质心
        centroids[batch_idx] = input_data[batch_idx, shuffled_indices[:n_clusters]]

    return centroids
input_data = torch.normal(0, 5, size=(batch_size, N)).cuda()

centroids = torch.zeros(batch_size, K).cuda()  # 随机生成质心数据，并将其移到GPU上
# initial_centroids = torch.zeros(batch_size, K).cuda()  # 可以根据需要定义初始质心
initial_centroids = torch.rand(batch_size, K).cuda()  # 可以根据需要定义初始质心
initial_centroids = random_initializer(input_data, K).cuda()
# 创建用于存储标签和输出的Tensor
labels = torch.zeros(batch_size, N, dtype=torch.int32).cuda()
output = torch.zeros(batch_size, N).cuda()

# print(input_data.dtype, output.dtype, input_data.device, output.device)
# exit(0)
# 调用你的CUDA函数
output_tensor = kmeans_parallel.kmeans_cuda_forward(input_data, centroids, labels, output, initial_centroids, batch_size, N, K)

print(output_tensor[0], output_tensor.shape)
print(initial_centroids[0], input_data[0], labels[0])

mse = nn.MSELoss()
print(f"mse: {mse(input_data, output_tensor)}")

# 打印输出结果（示例）
# print("Labels:")
# print(labels)

# print("Output:")
# print(output_tensor)