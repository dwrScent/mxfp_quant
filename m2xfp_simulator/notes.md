# notes
- rtl 论文展示：pe、quant engine面积，以及sram buffer

- accel_model_configs: 对应 q, k, v, o, mlp 三层

- 精度实验
    - baseline: olive, ant, mant
    - 自动化脚本：先设定一个精度，跑出结果再调
    - 设置只跑200 sample

- m2xfp 如何接入
    - accel_model_configs.py 定义 m2xfp 的 layer-wise bit pattern
    - conf_m2xfp.ini 定义 m2xfp 的阵列、带宽、buffer、精度范围
    - configs/ppa/*.csv 提供 m2xfp 对应 PE 的 area/power 数据

- benchmarks meaning
    - time: cycles
    - static: dram static/leakage energy
    - dram: dram dynamic read/write energy
    - buffer: sram WBUF/IBUF/OBUF dynamic energy
        - come from cacti sweep
    - core: core energy
        - come from ppa csvs

- DRAM(HBM): 片外，大，慢，能耗高
    - 存储KV cache，权重，较大的激活，输出
- SRAM: 片内，小，快，能耗低，加速器core附近
    - WBUF: 权重，通常固定(weight stationary)，多次复用
    - IBUF: 激活，层与层之间流动
    - OBUF: 输出，需要累加，可能高精度

## 修改
- 最小改动
    - 主gemm cycle 保持现有
    - metadata/scale bits -> DRAM/Buffer accesses
    - quant engine energy

- 后续开发
    - 更新dram
    - 支持端到端 

- benchmarks/：要模拟什么
- graph,tensor,tensorOps/：怎么把它表示出来
- simulator/：表示出来以后怎么算代价

