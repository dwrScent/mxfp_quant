# notes
- benchmarks/：要模拟什么
- graph,tensor,tensorOps/：怎么把它表示出来
- simulator/：表示出来以后怎么算代价

- m2xfp 如何接入
    - accel_model_configs.py 定义 m2xfp 的 layer-wise bit pattern
    - conf_m2xfp.ini 定义 m2xfp 的阵列、带宽、buffer、精度范围
    - configs/ppa/*.csv 提供 m2xfp 对应 PE 的 area/power 数据

- nvesm2

