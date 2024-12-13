import torch

def calculate_max_error(tensor_value, tensor_deq, q_group_size=-1):
    if q_group_size > 0:
        tensor_value = tensor_value.reshape(-1, q_group_size)
        tensor_deq = tensor_deq.reshape(-1, q_group_size)

    max_val = tensor_value.abs().amax(dim=1, keepdim=True)
    max_val_deq = tensor_deq.abs().amax(dim=1, keepdim=True)

    error_mean = (max_val - max_val_deq).abs().mean()
    error_max = (max_val - max_val_deq).abs().max()
    error_min = (max_val - max_val_deq).abs().min()

    relative_error = (max_val - max_val_deq).abs() / (max_val + 1e-8)  # 防止除以零
    relative_error_mean = relative_error.mean()
    relative_error_max = relative_error.max()
    relative_error_min = relative_error.min()

    mse_error = ((max_val - max_val_deq) ** 2).mean()

    print(f'Abs Error, mean: {error_mean}, max: {error_max}, min: {error_min}, MSE: {mse_error}')
    print(f'Relative Error, mean: {relative_error_mean}, max: {relative_error_max}, min: {relative_error_min}')


@torch.no_grad()
def mxfp_sub_group(tensor_value, quant_grid, sub_group_grid, mode="int", zero_point=True, q_group_size=-1, alpha=1.0, pos_value=None, get_labels=False, is_input=False, keep_outlier=False):
    assert torch.isnan(tensor_value).sum() == 0
    org_shape = tensor_value.shape
    quant_grid = quant_grid.to(tensor_value.device)

    sub_group_size = 2

    if q_group_size > 0:
        assert org_shape[-1] % q_group_size == 0
        tensor_value = tensor_value.reshape(-1, q_group_size)

    max_val = tensor_value.abs().amax(dim=1, keepdim=True)

    max_quant_val = max(quant_grid)
        
    # Compute the scaling factor
    exp = torch.floor(torch.log2(max_val)) - torch.floor(torch.log2(max_quant_val))
    scales = torch.pow(2, exp)

    # exp_max_val = torch.floor(torch.log2(max_val))
    # mask = torch.where(tensor_value > torch.pow(2, exp_max_val), torch.tensor(1), torch.tensor(0))

    zeros = 0

    outlier_mask = torch.zeros_like(tensor_value, dtype=torch.bool).to(tensor_value.device)
    _, indices = torch.topk(tensor_value.abs(), 1)
    outlier_mask.scatter_(1, indices, 1)

    outlier_group_mask = outlier_mask.reshape(-1, sub_group_size).to(dtype=torch.int8)
    outlier_group_mask = outlier_group_mask.sum(dim=1, keepdim=True)

    outlier_group_mask = outlier_group_mask.repeat(1, sub_group_size)
    outlier_group_mask = outlier_group_mask.reshape(-1, q_group_size)
    
    # print(outlier_mask, outlier_mask.shape, outlier_mask.sum(), outlier_group_mask, outlier_group_mask.shape, outlier_group_mask.sum())
    # exit(0)

    # Batch processing to avoid OOM
    batch_num = 4
    assert tensor_value.shape[0] % batch_num == 0
    batch_size = tensor_value.shape[0] // batch_num
    tensor_deq = torch.zeros_like(tensor_value)
    for idx in range(batch_num):
        tensor_par = tensor_value[idx*batch_size : (idx+1)*batch_size, :]
        labels = (((tensor_par + zeros) / scales[idx*batch_size : (idx+1)*batch_size, :]).unsqueeze(-1) - quant_grid).abs().argmin(dim=-1)
        tensor_q_par = quant_grid[labels] * scales[idx*batch_size : (idx+1)*batch_size, :] - zeros
        tensor_deq[idx*batch_size : (idx+1)*batch_size, :] = tensor_q_par
    
    tensor_deq_o_group = torch.zeros_like(tensor_value)
    for idx in range(batch_num):
        tensor_par = tensor_value[idx*batch_size : (idx+1)*batch_size, :]
        labels = (((tensor_par + zeros) / scales[idx*batch_size : (idx+1)*batch_size, :]).unsqueeze(-1) - sub_group_grid).abs().argmin(dim=-1)
        tensor_q_par = sub_group_grid[labels] * scales[idx*batch_size : (idx+1)*batch_size, :] - zeros
        tensor_deq_o_group[idx*batch_size : (idx+1)*batch_size, :] = tensor_q_par

    tensor_deq = tensor_deq * (1-outlier_group_mask) + tensor_deq_o_group * outlier_group_mask

    # tensor_deq = org_value * mask + tensor_deq * (1-mask)
    quant_mse = (tensor_deq-tensor_value).abs().pow(2).to(torch.float32)
    quant_mse_sum = torch.mean(quant_mse, dim=1, keepdim=True)

    # calculate_max_error(tensor_value, tensor_deq, q_group_size=q_group_size)
    # print(quant_mse, quant_mse_sum, quant_mse_sum.mean())
    # exit(0)

    assert torch.isnan(tensor_deq).sum() == 0
    assert torch.isnan(scales).sum() == 0
    assert torch.isnan(quant_mse).sum() == 0

    tensor_deq = tensor_deq.reshape(org_shape)

    quant_obj = 'input' if is_input else 'weight'
    print(f"Quantization MSE: {quant_mse_sum.mean().item()}, quant_obj: {quant_obj}, keep_outlier: {keep_outlier}")
    # print('init', scales, tensor_deq, tensor_value)

    if get_labels:
        return tensor_deq, quant_mse_sum, labels, quant_grid * scales
    else:
        return tensor_deq, quant_mse_sum

if __name__ == "__main__":
    tensor_value = torch.normal(0, 1, size=(4096, 4096))
    float_grid = [0.0, -0.0, 0.5, -0.5, 1.0, -1.0, 1.5, -1.5, 2.0, -2.0, 3.0, -3.0, 4.0, -4.0, 6.0, -6.0]
    sub_group_grid = [0, -4.0, -4.5, -5.0, -5.5, -6.0, -6.5, -7.0, -7.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5]
    quant_grid = torch.tensor(float_grid)
    sub_group_grid = torch.tensor(sub_group_grid)
    mxfp_sub_group(tensor_value, quant_grid, sub_group_grid, q_group_size=32)
