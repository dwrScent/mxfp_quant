### codes

- quantizer.py substitude linear layers with quantized linear layers

- entry.py parse_mxfp_modes() explains how to parse MXFP modes.

- qmodule_mxfp.py
    - sub_group_em() adding em_bits to sub-groups
    - quantize_data() return quantized data according to different methods ( or tensor_deq, quant_mse_sum? )
    - quantize_data is called by forward, (which is called by model's forward function.)?

### questions
- extremely low bit quantization

- when comparing encoding design, is it enough to compare only MSE?

- kv cache, positional encoding

- is it efficient to adopt both metadata and awq, gptq, quarot methods ?

#### other researches

- mx+ : use exp bit as mantisa bits for outliers

- BRIDGING THE GAP BETWEEN PROMISE AND PERFORMANCE FOR MICROSCALING FP4
QUANTIZATION: 
    - nvfp neutralize traditional outlier mitigation techniques. 
    - mxfp power-of-two scaling suffers degradation
    - introduce MICRO-Rotated-GPTQ: 
        - make use of hadamard transforms
        - variant of GPTQ
