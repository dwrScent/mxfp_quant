import torch
import torch.nn as nn
import kmeans_parallel
from sklearn.cluster import KMeans


batch_size = 1024
N = 1024
K = 16

def random_initializer(input_data, n_clusters):
    batch_size, N = input_data.shape    # 行 + 列
    centroids = torch.empty(batch_size, n_clusters, device=input_data.device, dtype=input_data.dtype)

    for batch_idx in range(batch_size):
        # 随机打乱输入数据的索引
        shuffled_indices = torch.randperm(N, device=input_data.device)
        
        # 从打乱的索引中选择前K个作为初始质心
        centroids[batch_idx] = input_data[batch_idx, shuffled_indices[:n_clusters]]

    return centroids


def mean_div_average_index_initializer(input_data, n_clusters):
    # 将数据从小到大排序后，然后根据比例分成16个组，每组取中间数
    batch_size, N = input_data.shape
    centroids = torch.empty(batch_size, n_clusters, device=input_data.device, dtype=input_data.dtype)

    sorted_tensor = torch.sort(input_data, dim=1).values # 对行进行排序
    
    group_size = N // n_clusters

    for batch_idx in range(batch_size):
        init_arr = torch.empty(n_clusters, device=input_data.device, dtype=input_data.dtype)
        for index in range(n_clusters - 1):
            i = group_size * (2 * index + 1) // 2
            init_arr[index] = sorted_tensor[batch_idx, i]
        
        init_arr[n_clusters - 1] = sorted_tensor[batch_idx, ((n_clusters - 2) * group_size + N - 1) // 2]
        centroids[batch_idx] = init_arr

    return centroids


def mean_div_average_value_initializer(input_data, n_clusters):
    batch_size, N = input_data.shape
    centroids = torch.empty(batch_size, n_clusters, device=input_data.device, dtype=input_data.dtype)
    sorted_tensor = torch.sort(input_data, dim=1).values

    group_size = N // n_clusters

    for batch_idx in range(batch_size):
        init_arr = torch.empty(n_clusters, device=input_data.device, dtype=input_data.dtype)
        for index in range(n_clusters - 1):
            # print(sorted_tensor[batch_idx][index * group_size : (index + 1) * group_size])
            init_arr[index] = torch.mean(sorted_tensor[batch_idx][index * group_size : (index + 1) * group_size])
        
        init_arr[n_clusters - 1] = torch.mean(sorted_tensor[batch_idx][(n_clusters - 2) * group_size : ])
        centroids[batch_idx] = init_arr

    return centroids


def dynamic_div_initializer(input_data, num_clusters):
    batch_size, N =input_data.shape
    centroids = torch.empty(batch_size, num_clusters, device=input_data.device, dtype=input_data.dtype)
    sorted_tensor = torch.sort(input_data, dim=1).values

    
    for batch_idx in range(batch_size):
        row = sorted_tensor[batch_idx].cpu()
        np_arr = row.numpy()
        kmeans = KMeans(n_clusters=num_clusters, n_init='auto')
        kmeans.fit(np_arr.reshape(-1, 1))
        centers = kmeans.cluster_centers_.reshape(1, -1)
        centroids[batch_idx] = torch.from_numpy(centers[0]).to(device=input_data.device, dtype=input_data.dtype)

    return centroids


input_data = torch.normal(10, 5, size=(batch_size, N)).cuda()

# print(input_data)

centroids1 = torch.zeros(batch_size, K).cuda()  # 随机生成质心数据，并将其移到GPU上
centroids2 = torch.zeros(batch_size, K).cuda()  # 随机生成质心数据，并将其移到GPU上
centroids3 = torch.zeros(batch_size, K).cuda()
centroids4 = torch.zeros(batch_size, K).cuda()
# initial_centroids = torch.zeros(batch_size, K).cuda()  # 可以根据需要定义初始质心
initial_centroids = torch.rand(batch_size, K).cuda()  # 可以根据需要定义初始质心
initial_centroids1 = random_initializer(input_data, K).cuda()
initial_centroids2 = mean_div_average_index_initializer(input_data, K).cuda()
initial_centroids3 = mean_div_average_value_initializer(input_data, K).cuda()
initial_centroids4 = dynamic_div_initializer(input_data, K).cuda()
print(initial_centroids4, initial_centroids4.shape, input_data.shape)
exit(0)
# 创建用于存储标签和输出的Tensor
labels1 = torch.zeros(batch_size, N, dtype=torch.int32).cuda()
output1 = torch.zeros(batch_size, N).cuda()
labels2 = torch.zeros(batch_size, N, dtype=torch.int32).cuda()
output2 = torch.zeros(batch_size, N).cuda()
labels3 = torch.zeros(batch_size, N, dtype=torch.int32).cuda()
output3 = torch.zeros(batch_size, N).cuda()
labels4 = torch.zeros(batch_size, N, dtype=torch.int32).cuda()
output4 = torch.zeros(batch_size, N).cuda()

# print(input_data.dtype, output.dtype, input_data.device, output.device)
# exit(0)
# 调用你的CUDA函数
output_tensor1 = kmeans_parallel.kmeans_cuda_forward(input_data, centroids1, labels1, output1, initial_centroids1, batch_size, N, K)
output_tensor2 = kmeans_parallel.kmeans_cuda_forward(input_data, centroids2, labels2, output2, initial_centroids2, batch_size, N, K)
output_tensor3 = kmeans_parallel.kmeans_cuda_forward(input_data, centroids3, labels3, output3, initial_centroids3, batch_size, N, K)
output_tensor4 = kmeans_parallel.kmeans_cuda_forward(input_data, centroids4, labels4, output4, initial_centroids4, batch_size, N, K)
# print(output_tensor[0], output_tensor.shape)
# print(initial_centroids[0], input_data[0], labels[0])
counts1 = torch.bincount(labels1[128].flatten(), minlength=16)
counts2 = torch.bincount(labels2[128].flatten(), minlength=16)
counts3 = torch.bincount(labels3[128].flatten(), minlength=16)
counts4 = torch.bincount(labels4[128].flatten(), minlength=16)

mse = nn.MSELoss()
print(f"mse of old init: {mse(input_data, output_tensor1)}, counts: {counts1}")
print(f"mse of new init: {mse(input_data, output_tensor2)}, counts: {counts2}")
print(f"mse of mean value init : {mse(input_data, output_tensor3)}, counts: {counts3}")
print(f"mse of dynamic div init : {mse(input_data, output_tensor4)}, counts: {counts4}")
# 打印输出结果（示例）
# print("Labels:")
# print(labels)

# print("Output:")
# print(output_tensor)