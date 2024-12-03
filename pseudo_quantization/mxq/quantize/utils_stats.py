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