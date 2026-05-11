## time/static/dram/buffer/core 计算流程

### 总入口
- `run_simulator.py` 是最终入口。
- 对每个 accelerator：
    - `run_sim(accelerator)` 读取 `configs/accelerator/conf_{accelerator}.ini`
        - `if_width`: DRAM interface bandwidth，单位是 bits/cycle
        - `a`, `c`: systolic array 的 `N`, `M`
        - `high_prec`, `low_prec`: `pmax`, `pmin`
        - `Wgt_SRAM`, `Act_SRAM`, `Out_SRAM`: WBUF/IBUF/OBUF 容量，配置里是 Bytes
    - 选择 core PPA CSV：
        - `mant`: `configs/ppa/systolic_array_synth_mant.csv`
        - `nvesm2`: `configs/ppa/systolic_array_synth_nvesm2.csv`
        - `nvfp`: `configs/ppa/systolic_array_synth_nvfp.csv`
        - 其他默认用 `configs/ppa/systolic_array_synth.csv`
    - 创建 `Simulator(config_file, core_csv_path=...)`
    - `check_pandas_or_run(...)` 调 `SimulatorSweep.sweep(...)`
    - `benchmarks.get_bench_nn(...)` 用 `benchmarks/accel_model_configs.py` 的 bit pattern 和 `benchmarks/base_models.py` 的 shape 生成 graph
    - `benchmarks.get_bench_numbers(...)` 对 graph 中每个 op 调 `sim_obj.get_cycles(...)`
    - 每层得到一个 `Stats`：
        - `total_cycles`
        - `mem_stall_cycles`
        - `reads['act'/'wgt'/'out'/'dram']`
        - `writes['act'/'wgt'/'out'/'dram']`
    - `run_simulator.py` 先把所有 layer 按 `Network` sum，得到每个模型的总 cycles 和总 read/write bits
    - energy breakdown 由 `Stats.get_energy_breakdown(sim.get_energy_cost())` 计算

### time
- `time` 在最终 `results/m2xfp_res.csv` 中写成 `Time` 行，含义不是绝对时间，而是 normalized cycles：
    - 对每个模型：`accelerator_cycles / normalized_bench_cycles`
    - 最后一组 `Mean`：所有模型 ratio 的平均
- 原始 cycles 来自 `Stats.total_cycles`。
- 每层的 `Stats.total_cycles` 在 `accelerator/src/optimizer/optimizer.py:get_stats_fast(...)` 中计算：
    - 先枚举 tiling：
        - `B/b`
        - `OW/ow`
        - `OH/oh`
        - `IC/ic`
        - `OC/oc`
    - `num_tiles = num_b * num_ow * num_oh * num_ic * num_oc`
    - 默认 `run_simulator.py -> check_pandas_or_run(...)` 没传 `weight_stationary=True`，所以默认走 output-stationary compute cycles：
        - `ni = kw * kh * ic`
        - `no = oc`
        - `batch = b * oh * ow`
        - `perf(prec) = int(pmax / max(prec, pmin))`
        - 单 tile compute cycles：
            - `ceil(batch / (N * perf(iprec))) * ceil(no / (M * perf(wprec))) * ni`
        - 总 compute cycles：
            - `compute_cycles = num_tiles * tile_compute_cycles`
    - 如果显式使用 `weight_stationary=True`，单 tile compute cycles 换成：
        - `batch * ceil(no / (M * perf(wprec))) * ceil(ni / (N * perf(iprec)))`
- memory stall cycles：
    - optimizer 会估算 DRAM traffic，并把一部分 memory cycles 和 compute cycles overlap
    - `initial_dram_reads = sum(max_write_size[namespace])`
        - 初始把 wgt/act/out 等 tile 数据从 DRAM 读入 SRAM
    - `final_dram_writes = sum(max_read_size[namespace])`
        - 最后把结果从 SRAM 写回 DRAM
    - `latency = ceil(initial_dram_reads / if_width) + ceil(final_dram_writes / if_width)`
    - `total_dram_accesses = stats.reads['dram'] + stats.writes['dram']`
    - `middle_dram_accesses = total_dram_accesses - initial_dram_reads - final_dram_writes`
    - `memory_cycles_required = ceil(middle_dram_accesses / if_width)`
    - `memory_stalls = max(0, memory_cycles_required - compute_cycles) + latency`
    - `total_cycles = compute_cycles + memory_stalls`
    - `mem_stall_cycles = memory_stalls`
- 直观理解：
    - 中间 DRAM 传输可以和 compute overlap；只有超过 compute 可覆盖部分才形成 stall
    - 初始加载和最终写回通过 `latency` 固定计入 stall

### static
- `static` 在最终 CSV 中写成 `Static` 行。
- 它来自 `Stats.get_energy_breakdown(...)` 的第 1 个分量：
    - `dram_leak_energy = 484.615 / 500`
    - `static_energy = total_cycles * dram_leak_energy`
- 这里的 static 只统计 DRAM static/leakage energy。
- 注意：
    - `get_energy_breakdown(...)` 里没有把 SRAM leakage 算进 `Static`
    - `simulator.get_energy_cost(...)` 虽然会从 CACTI 得到 `sram_leak_energy`，但 breakdown 中对应代码被注释了：
        - `# sram_energy += self.total_cycles * energy_cost.sram_leak_energy`
- 最终写入 `results/m2xfp_res.csv` 时，`Static` 也不是绝对能耗，而是：
    - `current_accelerator_static_energy / normalized_bench_total_energy`
    - 其中 `normalized_bench_total_energy = Static + Dram + Buffer + Core`

### dram
- `dram` 在最终 CSV 中写成 `Dram` 行，表示 DRAM dynamic read/write energy。
- 它来自 `Stats.get_energy_breakdown(...)` 的第 2 个分量：
    - `dram_cost_read = 0.644304 / 1024`
    - `dram_cost_write = 0.784104 / 1024`
    - `dram_energy = reads['dram'] * dram_cost_read + writes['dram'] * dram_cost_write`
- `reads['dram']` 和 `writes['dram']` 的单位是 bits。
- DRAM traffic 在 `get_stats_fast(...)` 中由 SRAM tile 搬运推导：
    - 每次需要把数据写入片上 buffer 时，等价于从 DRAM 读：
        - `stats.reads['dram'] += writes[namespace]`
    - 每次需要从片上 buffer 读出最终结果时，等价于向 DRAM 写：
        - `stats.writes['dram'] += reads[namespace]`
- 最终 `Dram` 行的归一化方式：
    - `current_accelerator_dram_energy / normalized_bench_total_energy`

### buffer
- `buffer` 在最终 CSV 中写成 `Buffer` 行，表示 SRAM WBUF/IBUF/OBUF dynamic energy。
- 它来自 `Stats.get_energy_breakdown(...)` 的第 3 个分量：
    - WBUF:
        - `reads['wgt'] * energy_cost.wbuf_read_energy`
        - `writes['wgt'] * energy_cost.wbuf_write_energy`
    - IBUF:
        - `reads['act'] * energy_cost.ibuf_read_energy`
        - `writes['act'] * energy_cost.ibuf_write_energy`
    - OBUF:
        - `reads['out'] * energy_cost.obuf_read_energy`
        - `writes['out'] * energy_cost.obuf_write_energy`
- `reads[...]` 和 `writes[...]` 的单位是 bits。
- `energy_cost.*buf_*_energy` 来自 `Simulator.get_energy_cost(...)`：
    - 用 SRAM 容量、bank 数、block width 组成 CACTI query
    - `wbuf_bank = 32`
    - `ibuf_bank = 32`
    - `obuf_bank = 32 * 32`
    - `wbuf_bits = 16 * 32`
    - `ibuf_bits = 16 * 32`
    - `obuf_bits = 32`
    - 每个 buffer 的 per-bit energy：
        - `read_energy_nJ / buffer_access_bits`
        - `write_energy_nJ / buffer_access_bits`
- SRAM access count 的来源：
    - 外层 tile promotion 决定片上 buffer 需要被填充/保留多少数据
    - 内层 compute 根据 dataflow 累加 WBUF/IBUF/OBUF read/write
    - 默认 output-stationary：
        - `IBUF read += num_tiles * (oc * oh * ow * b) * (kw * kh * ic) * iprec`
        - `WBUF read += num_tiles * (oc * oh * ow * b) * (kw * kh * ic) * wprec`
        - `OBUF read += num_tiles * (oc * oh * ow * b) * oprec`
        - `OBUF write += num_tiles * (oc * oh * ow * b) * oprec`
    - `oprec` 在 `get_stats_fast(...)` 的内层统计里是 16
- 最终 `Buffer` 行的归一化方式：
    - `current_accelerator_buffer_energy / normalized_bench_total_energy`

### core
- `core` 在最终 CSV 中写成 `Core` 行，表示 systolic array core energy。
- 它来自 `Stats.get_energy_breakdown(...)` 的第 4 个分量：
    - `core_energy = total_cycles * energy_cost.core_leak_energy`
    - `core_energy += (total_cycles - mem_stall_cycles) * energy_cost.core_dynamic_energy`
- 含义：
    - core leakage 按总 cycles 计
    - core dynamic 只按非 memory-stall cycles 计，即 compute active cycles：
        - `active_cycles = total_cycles - mem_stall_cycles`
- `energy_cost.core_*` 来自 `Simulator.get_energy_cost(...)` 读取 core PPA CSV：
    - 用 `pmax`, `pmin`, `N`, `M` 查表
    - 如果 CSV 里没有完整 `N,M` 阵列行，就 fallback 到 `N=1,M=1` 的单 PE 行，然后乘以 `N * M`
    - `core_dynamic_energy = Dynamic Power (nW) / Frequency`
    - `core_leak_energy = Leakage Power (nW) / Frequency`
- 最终 `Core` 行的归一化方式：
    - `current_accelerator_core_energy / normalized_bench_total_energy`

### 五个输出项之间的关系
- 原始每个模型：
    - `total_energy = Static + Dram + Buffer + Core`
- `Time` 的 denominator 是 baseline cycles：
    - `Time_ratio = accelerator_cycles / baseline_cycles`
- `Static/Dram/Buffer/Core` 的 denominator 都是 baseline total energy：
    - `Component_ratio = accelerator_component_energy / baseline_total_energy`
- 所以同一个 accelerator 的四个 energy component ratio 加起来，才是该 accelerator 相对 baseline 的 total energy ratio：
    - `(Static + Dram + Buffer + Core) / baseline_total_energy`

