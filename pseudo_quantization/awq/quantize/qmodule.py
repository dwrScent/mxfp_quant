import math
import torch
import torch.nn as nn

def make_divisible(c, divisor):
    return (c + divisor - 1) // divisor


def calculate_zeros_width(in_features, group_size=128, pack_num=8):
    if group_size >= 128:
        size_multiplier = 1
    elif group_size == 64:
        size_multiplier = 2
    elif group_size == 32:
        size_multiplier = 4
    else:
        raise NotImplementedError
    
    base_width = make_divisible(in_features // group_size, pack_num)
    base_width = make_divisible(base_width, size_multiplier) * size_multiplier
    return base_width

def reorder_interleave(weight, codebook):
    _, entry_size = codebook.shape
    reordered_weight = torch.empty_like(weight)
    reordered_codebook = torch.empty_like(codebook)
    
    # 遍历每一行进行重排序
    for i in range(weight.shape[0]):
        # 获取当前行的权重
        row_weight = weight[i]
        # 计算当前行每个 index 的出现次数
        freqs = torch.bincount(row_weight, minlength=entry_size)
        freq_sorted_indices = torch.argsort(freqs, descending=True)
        # 创建一个新的索引映射，频率最高的一半放在偶数位，频率较低的一半放在奇数位
        new_indices = torch.empty(entry_size, dtype=torch.long)
        new_indices[::2] = freq_sorted_indices[:entry_size // 2]
        new_indices[1::2] = freq_sorted_indices[entry_size // 2:]
        # 反向映射，用于重排序 weight
        index_mapping = torch.argsort(new_indices).to(weight.device)
        # 重新排序 weight
        reordered_weight[i] = index_mapping[row_weight]
        # 重新排序 codebook 的当前行
        reordered_codebook[i] = codebook[i][new_indices]
    return reordered_weight, reordered_codebook

class ScaledActivation(nn.Module):
    def __init__(self, module, scales):
        super().__init__()
        self.act = module
        self.scales = nn.Parameter(scales.data)
    
    def forward(self, x):
        return self.act(x) / self.scales.view(1, 1, -1).to(x.device)

class WQLinear(nn.Module):
    def __init__(self, w_bit, group_size, in_features, out_features, bias, dev):
        super().__init__()
        
        if w_bit not in [4, 5]:
            raise NotImplementedError("Only 4-bit and 5-bit are supported for now.")
        
        self.in_features = in_features
        self.out_features = out_features
        self.w_bit = w_bit
        self.group_size = group_size 
        pack_group = group_size if group_size != -1 else in_features
        self.split_k_iters = 1
        # quick sanity check (make sure aligment)
        assert self.in_features % self.group_size == 0
        # assert out_features % (32 // self.w_bit) == 0

        # 32-bit 存放 8 个 4-bit
        pack_num = 32 // 4
        pack_num_suffix = 32 if w_bit == 5 else None
        
        # TODO (Haotian): a function for buffer shape calculation
  
        self.register_buffer('qweight', torch.zeros((out_features, in_features // pack_num), dtype=torch.int32, device=dev))
        self.register_buffer('codebook', torch.zeros((out_features, in_features // pack_group * (2 ** w_bit)), dtype=torch.float16, device=dev))
        # self.register_buffer('scales', torch.zeros((out_features, calculate_zeros_width(in_features, self.group_size) * pack_num), dtype=torch.float16, device=dev))

        if self.w_bit == 5:
            self.register_buffer('qweight_suffix', torch.zeros((out_features, in_features // pack_num_suffix), dtype=torch.int32, device=dev))
        if bias:
            self.register_buffer('bias', torch.zeros((out_features), dtype=torch.float16, device=dev))
        else:
            self.bias = None

    @classmethod
    def from_linear(cls, linear, w_bit, group_size, init_only=False, labels=None, codebook=None):
        awq_linear = cls(w_bit, group_size, linear.in_features, linear.out_features, linear.bias is not None, linear.weight.device)
        if init_only:  # just prepare for loading sd
            return awq_linear

        # 32-bit 存放 8 个 4-bit
        pack_num = 32 // 4
        pack_num_suffix = 32 if w_bit == 5 else None

        if linear.bias is not None:
            awq_linear.bias = linear.bias.clone().half()
        
        use_reordered = True if w_bit == 5 else False
        # use_reordered = False
        if use_reordered:
            reordered_labels, reordered_codebook = reorder_interleave(labels, codebook)
            rows = torch.arange(codebook.size(0)).unsqueeze(1).expand_as(labels)
            reordered_rows = torch.arange(reordered_codebook.size(0)).unsqueeze(1).expand_as(reordered_labels)
    
            weight = codebook[rows, labels]
            reordered_weight = reordered_codebook[reordered_rows, reordered_labels]
            assert torch.equal(weight, reordered_weight)
            intweight = reordered_labels.to(dtype=torch.int32)
            awq_linear.codebook = reordered_codebook.reshape(linear.out_features, -1) 
        else:
            intweight = labels.to(dtype=torch.int32)
            awq_linear.codebook = codebook.reshape(linear.out_features, -1) 


        qweight = torch.zeros((intweight.shape[0], intweight.shape[1] // pack_num), dtype=torch.int32, device=intweight.device)
        if awq_linear.w_bit == 5:
            qweight_suffix = torch.zeros((intweight.shape[0], intweight.shape[1] // pack_num_suffix), dtype=torch.int32, device=intweight.device)   
        
        pack_offset = 1 if awq_linear.w_bit == 5 else 0
        for col in range(intweight.shape[1] // pack_num):
            # order_map = [0, 2, 4, 6, 1, 3, 5, 7]
            order_map = [0, 1, 2, 3, 4, 5, 6, 7]
            # order_map = [7, 6, 5, 4, 3, 2, 1, 0]
            for i in range(pack_num):
                # 直接提取出所有列，这样写的目的是所有 input channel 并行 pack；
                # 对于 5-bit，先取出 top 4-bit；对于 4-bit，直接取出 4-bit
                qweight_col = (intweight[:, col * pack_num + order_map[i]] >> pack_offset)
                # print(qweight_col, qweight_col.shape, i * awq_linear.w_bit)
                # qwegiht shape: (out_features, in_features // pack_num), (4096, 512)
                # 将 8 个元素 pack 到 qweight 中的 1 个元素中
                qweight[:, col] |= qweight_col << (i * 4)

        if awq_linear.w_bit == 5:
            for col in range(intweight.shape[1] // pack_num_suffix):
                for i in range(pack_num_suffix):
                    # 对于 5-bit，取出 bottom 1-bit
                    qweight_col = (intweight[:, col * pack_num_suffix + i] & 1)
                    qweight_suffix[:, col] |= qweight_col << (i * 1)
        # print(intweight, qweight, qweight.shape, qweight_suffix, qweight_suffix.shape, codebook, codebook.shape)
        # torch.set_printoptions(profile="full")
        # print(intweight[0], qweight[0], qweight_suffix[0])
        # exit(0)
        # print(intweight, qweight, qweight.shape)
        # exit(0)
                    
        awq_linear.qweight = qweight.reshape(linear.out_features, -1)
        if awq_linear.w_bit == 5:
            awq_linear.qweight_suffix = qweight_suffix.reshape(linear.out_features, -1)
        
        if use_reordered:
            init_bit = 0
            compressed_bit = 0
            for i in range(qweight_suffix.shape[0]):
            # for i in range(1):
                flat_channel = qweight_suffix[i].flatten().cpu()
                bytes_array = flat_channel.numpy().tobytes()
                bit_stream = ''.join(f'{byte:08b}' for byte in bytes_array)
   
                # compressed_stream, uncompressed_segments, compressed_segments = compress_segments(bit_stream, 32)
                # if len(compressed_stream) > len(bit_stream):
                #     compressed_stream = bit_stream
                # init_bit += len(bit_stream)
                # compressed_bit += len(compressed_stream)

            # print(bit_stream, num_zeros, num_ones, (num_zeros / (num_ones + num_zeros)))
            # exit(0)
            # print(init_bit, compressed_bit, compress_ratio)
                
            # compress_ratio = compressed_bit / init_bit
            # print(f"{init_bit}, {compressed_bit}, {compress_ratio}")
            flat_tensor = qweight_suffix.flatten().cpu()
            bytes_array = flat_tensor.numpy().tobytes()
            bit_stream = ''.join(f'{byte:08b}' for byte in bytes_array)
            num_zeros = bit_stream.count('0')
            num_ones = bit_stream.count('1')
            print(f"{num_zeros}, {num_ones}, {num_zeros / num_ones}, {num_zeros / (num_ones + num_zeros)}")

        # 如果 qweight 数值是负数，那么尾数部分需要取反+1 才能看到正确二进制数，原因是负数的二进制表示是补码表示
        # print(qweight, qweight.shape, intweight, intweight.shape, qweight[0][0] & 0x0000000f, (qweight[0][0] & 0x000000f0) >> 4)
        # exit(0)

        return awq_linear

    @torch.no_grad()
    def forward(self, x):
        out_shape = x.shape[:-1] + (self.out_features, )
        out = awq_inference_engine.gemm_forward_cuda(x.reshape(-1, x.shape[-1]), self.qweight, self.scales, self.qzeros, 8)
        out = out + self.bias if self.bias is not None else out
        return out.reshape(out_shape)
    