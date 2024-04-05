import torch
import torch.nn as nn
import torch.nn.functional as F
import math


@torch.no_grad()
def pseudo_quantize_tensor_mokey(w, outlier_ratio = 0.015, outlier_dict=None, check=False):
    from sklearn.cluster import KMeans
    mokey_flag = True
    golden_cb = torch.tensor([ 0.2020, -0.2020,  0.4131, -0.4131,  0.6616, -0.6616,  0.9551, -0.9551, 1.3008, -1.3008,  1.7090, -1.7090,  2.1895, -2.1895, 0]).to(w.device).to(torch.half)
    int_cb = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 0, -1.0, -2.0, -3.0, -4.0, -5.0, -6.0, -7.0]).to(w.device).to(torch.half)
    if len(w.shape) == 2:
        ic = w.shape[1]
        oc = w.shape[0]
    else:
        ic = w.shape[1]
        oc = w.shape[2]
    # print(w.shape, flush=True)
    
    org_w_shape = w.shape
    w = w.reshape(ic * oc, 1)
    
    tensor_sum = w.to(torch.float64).sum()
    tensor_square_sum = w.to(torch.float64).pow(2).sum()
    # Mokey : tensor-wise quantization
    # start_time = time.time()
    if mokey_flag:
        num_to_keep = math.ceil(ic * oc * outlier_ratio)
        outlier_val, _ = torch.topk(w.abs(), num_to_keep, dim=0)
        outlier_num = outlier_val.shape[0]
        outlier_sum = outlier_val.to(torch.float64).sum()
        outlier_square_sum = outlier_val.to(torch.float64).pow(2).sum()
        outlier_mask = (w.abs() >= outlier_val.abs().min()).to(torch.int32)

        non_outlier_mask = 1 - outlier_mask
        # print(time.time() - start_time, flush=True)
        # mask weight outliers
        outliers = w * outlier_mask
        masked_w = w * non_outlier_mask
        if check == True:
            print("check w_mask")
            print(tensor_square_sum)
            print(outlier_square_sum)
            print(tensor_sum)
            print(outlier_sum)
            print(masked_w, flush=True)
            
    
        mean = ((tensor_sum - outlier_sum) / (ic * oc - outlier_num)).to(torch.half)
        std_2 = (tensor_square_sum - outlier_square_sum) / (ic * oc - outlier_num) - mean.pow(2)
        std = std_2.sqrt().to(torch.half)
        w_deq = golden_cb[(((masked_w.unsqueeze(-1) - mean) / std) - golden_cb).abs().argmin(dim=-1)] * std + mean
        
        # outlier processing
        if outlier_dict is None:
            kmeans = KMeans(n_clusters=16, init="k-means++", n_init=1, max_iter=300)
            
            # print(masked_w.shape)
            # print(outliers.shape)
            
            # print(outlier_val.squeeze(-1).to("cpu").numpy())
            # exit()

            outlier_val = torch.masked_select(w, outlier_mask.to(torch.bool))

            # print(w, flush=True)
            X = kmeans.fit_predict(outlier_val.squeeze(-1).to("cpu").numpy().reshape(-1, 1))

            outlier_centroids = torch.from_numpy(kmeans.cluster_centers_).to(w.device).to(torch.half)
            # print(golden_cb.shape)
            # print(outlier_centroids.shape)

            outlier_deq = outlier_centroids[(outliers - outlier_centroids.squeeze(-1)).abs().argmin(dim=-1)]
            
        else:
            outlier_deq = outlier_dict[(outliers - outlier_dict.squeeze(-1)).abs().argmin(dim=-1)]
        # print(outliers.shape)
        # print(outlier_deq.shape)
        # print(w_deq.shape)
        # print(non_outlier_mask.shape)
        w = w_deq * non_outlier_mask + outlier_deq * outlier_mask

    
    else:
    # max_val = masked_w.abs().max()
        max_val = w.abs().max()
        max_cb = int_cb.max()
        
        scale = max_val / max_cb


        w_deq = int_cb[((w.unsqueeze(-1) / scale) - int_cb).abs().argmin(dim=-1)] * scale
        w = w_deq
    

    w = w.reshape(org_w_shape)
    
    if outlier_dict is None:
        return w, outlier_centroids
    else:
        return w, outlier_dict
    
    
class Mokey_Linear(nn.Module):
    def __init__(self, in_features, out_features, layer_id, layer_name, dev):
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        
        self.layer_id = layer_id
        self.layer_name = layer_name
        
        self.act_outlier_dict = None
        
        self.register_buffer('weight', torch.zeros((out_features, in_features), dtype=torch.float16, device=dev))
        
        
    
    @classmethod
    def from_linear(cls, linear, layer_id, layer_name, init_only=False):
        mokey_linear = cls(linear.in_features, linear.out_features, layer_id, layer_name, linear.weight.device)
        if init_only:
            return mokey_linear
        
        mokey_linear.weight = linear.weight.data.clone().half()
        return mokey_linear
    
    @torch.no_grad()
    def forward(self, x):
        out_shape = x.shape[:-1] + (self.out_features, )
        
        input = x.reshape(-1, x.shape[-1])

        if self.act_outlier_dict is None:
            deq_weight, _ = pseudo_quantize_tensor_mokey(self.weight, outlier_ratio=0.015, outlier_dict=None)
            self.weight = deq_weight
            # self.act_outlier_dict = [0]
            if self.layer_id == 1:
                check = True
            else:
                check = True
            deq_input, self.act_outlier_dict = pseudo_quantize_tensor_mokey(input, outlier_ratio=0.045, outlier_dict=None, check=check)
        else:
            deq_input, _ = pseudo_quantize_tensor_mokey(input, outlier_ratio=0.045, outlier_dict=self.act_outlier_dict)
            
        # print(deq_input)
        out = F.linear(deq_input, self.weight)
        

            # print(input)
            # print("deq_input")
            # print(deq_input)
            # print(self.layer_name, flush=True)
        
        # print(out_shape)
        # print(out, flush=True)
        return out.reshape(out_shape)
        