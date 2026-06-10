import argparse
import pandas
import os
import numpy as np
import benchmarks.benchmarks as benchmarks
from accelerator.src.simulator.stats import Stats
from accelerator.src.simulator.simulator import Simulator
from accelerator.src.sweep.sweep import SimulatorSweep, check_pandas_or_run
from accelerator.src.utils.utils import *

def df_to_stats(df):
    stats = Stats()
    stats.total_cycles = float(df['Cycles'])
    stats.mem_stall_cycles = float(df['Memory wait cycles'])
    stats.reads['act'] = float(df['IBUF Read'])
    stats.reads['out'] = float(df['OBUF Read'])
    stats.reads['wgt'] = float(df['WBUF Read'])
    stats.reads['dram'] = float(df['DRAM Read'])
    stats.writes['act'] = float(df['IBUF Write'])
    stats.writes['out'] = float(df['OBUF Write'])
    stats.writes['wgt'] = float(df['WBUF Write'])
    stats.writes['dram'] = float(df['DRAM Write'])
    return stats

sim_sweep_columns = ['N', 'M',
        'Max Precision (bits)', 'Min Precision (bits)',
        'Network', 'Layer',
        'Cycles', 'Memory wait cycles',
        'WBUF Read', 'WBUF Write',
        'OBUF Read', 'OBUF Write',
        'IBUF Read', 'IBUF Write',
        'DRAM Read', 'DRAM Write',
        'Bandwidth (bits/cycle)',
        'WBUF Size (bits)', 'OBUF Size (bits)', 'IBUF Size (bits)',
        'Batch size']

# batch_size = 64
# batch_size = 512
batch_size = 1
# batch_size = 16

# directory to store the .csv
results_dir = './results'
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

bf_e_cycles = {}
bf_e_energy = {}

model_name_dict = {}          # will be filled based on benchmarks.benchlist
accelerator_list = []          # will be set from CLI
normalized_bench = ''    # default baseline accelerator; can be overridden

def run_sim(accelerator):
    # Get the configuration file for the given benchmark type
    config_file = f'configs/accelerator/conf_{accelerator}.ini'

    core_csv_paths = {
        'ant': 'configs/ppa/systolic_array_synth_ant.csv',
        'olive': 'configs/ppa/systolic_array_synth_olive.csv',
        'mant': 'configs/ppa/systolic_array_synth_mant.csv',
        'microscopiq': 'configs/ppa/systolic_array_synth_microscopiq.csv',
        'm2xfp': 'configs/ppa/systolic_array_synth_m2xfp.csv',
        'nvesm2': 'configs/ppa/systolic_array_synth_nvesm2.csv',
    }
    if accelerator not in core_csv_paths:
        raise ValueError(f"No core PPA CSV configured for accelerator: {accelerator}")
    core_csv_path = core_csv_paths[accelerator]


    # Create simulator object
    bf_e_sim = Simulator(config_file, False, core_csv_path=core_csv_path)
    bf_e_sim_sweep_csv = os.path.join(results_dir, f'{accelerator}.csv')
    bf_e_sim_sweep_df = pandas.DataFrame(columns=sim_sweep_columns)
    bf_e_results = check_pandas_or_run(bf_e_sim, bf_e_sim_sweep_df, bf_e_sim_sweep_csv, batch_size=batch_size, accelerator=accelerator)
    bf_e_results = bf_e_results.groupby('Network',as_index=False).agg(np.sum)

    # Store the total cycles and energy for each network
    bf_e_cycles[accelerator] = []
    bf_e_energy[accelerator] = []
    for name in benchmarks.benchlist:
        bf_e_stats = df_to_stats(bf_e_results.loc[bf_e_results['Network'] == name])
        bf_e_cycles[accelerator].append(bf_e_stats.total_cycles)
        bf_e_energy[accelerator].append(bf_e_stats.get_energy_breakdown(bf_e_sim.get_energy_cost()))
    
    # Print the cycle and energy results for this benchmark type
    print(f"{accelerator} cycle", bf_e_cycles[accelerator])
    print(f"{accelerator} energy", bf_e_energy[accelerator])


def process_result():
    # Use global normalized_bench, accelerator_list, model_name_dict, and benchmarks.benchlist
    ENERGY_COMPONENTS = ['Static', 'Dram', 'Buffer', 'Core']

    num_models = len(bf_e_cycles[normalized_bench])    
    num_accelerators = len(accelerator_list)

    with open(os.path.join(os.getcwd(), 'results', 'm2xfp_res.csv'), "a") as ff:
        wr_stats_line = "Time, "
        wr_bench_name = ", "
        wr_model_name = ", "

        # ========================
        # 1) Cycle stats
        # ========================

        # Use the globally chosen normalized_bench
        tmp_cycle = {}
        tmp_cycle_mean = {}

        # Initialize Mean cycles
        for accelerator in bf_e_cycles:
            tmp_cycle_mean[accelerator] = 0
        all_cyc = []

        normalized_cycle = bf_e_cycles[normalized_bench]
        for i in range(num_models):
            model_name = benchmarks.benchlist[i]
            for accelerator, cycles in bf_e_cycles.items():
                tmp_cycle[accelerator] = cycles[i] / normalized_cycle[i]
                tmp_cycle_mean[accelerator] += tmp_cycle[accelerator]

                all_cyc.append(tmp_cycle[accelerator])
                wr_bench_name += f"{accelerator}, "
                wr_stats_line += "%0.5f, " % (tmp_cycle[accelerator])
            # wr_model_name += f"{model_name_dict[model_name]}, , , , , , "
            wr_model_name += f"{model_name_dict[model_name]}, " + ", " * (num_accelerators - 1)

        # Process and write Mean
        for accelerator, cycles in bf_e_cycles.items():
            tmp_cycle_mean[accelerator] /= num_models
            wr_bench_name += f"{accelerator}, "
            wr_stats_line += "%0.5f, " % (tmp_cycle_mean[accelerator])

        # wr_model_name += "Mean, , , , , \n"
        wr_model_name += "Mean, " + ", " * (num_accelerators - 1) + "\n"
        wr_bench_name += "\n"
        wr_stats_line += "\n"
        ff.write(wr_model_name)
        ff.write(wr_bench_name)
        ff.write(wr_stats_line)

        # ========================
        # 2) Energy stats
        # ========================

        # all_energy[component][model_name] = [ratio_for_accelerator_0, ..., ratio_for_accelerator_{K-1}]
        all_energy = {comp: {} for comp in ENERGY_COMPONENTS}

        # energy_sum[component][accelerator] = sum over models (后面算平均用)
        energy_sum = {
            comp: {accelerator: 0.0 for accelerator in accelerator_list}
            for comp in ENERGY_COMPONENTS
        }

        # 先按模型把归一化后的 energy ratio 算出来
        for i in range(num_models):
            model_name = benchmarks.benchlist[i]

            # baseline（normalized_bench）在该模型上的 total energy
            norm_total = sum(bf_e_energy[normalized_bench][i])

            # 为当前模型初始化 per-component 列表
            for comp in ENERGY_COMPONENTS:
                all_energy[comp][model_name] = [0.0 for _ in accelerator_list]

            # 遍历每个 accelerator，按 component 填写数值
            for b_idx, accelerator in enumerate(accelerator_list):
                # bf_e_energy[accelerator][i] 是一个长度为 4 的 list: [Static, Dram, Buffer, Core]
                comp_vals = bf_e_energy[accelerator][i]
                for comp_idx, comp in enumerate(ENERGY_COMPONENTS):
                    ratio = comp_vals[comp_idx] / norm_total
                    all_energy[comp][model_name][b_idx] = ratio
                    energy_sum[comp][accelerator] += ratio

        # 复用上面写过的 header（model names + bench types）
        ff.write(wr_model_name)
        ff.write(wr_bench_name)

        # 对每个 component 写一行：所有模型的 ratio + 最后的平均值
        for comp in ENERGY_COMPONENTS:
            wr_stats_line = f"{comp}, "
            # 1) 先按模型顺序写出各 accelerator 的 ratio
            for model_name in benchmarks.benchlist:
                values = all_energy[comp][model_name]  # list over accelerator_list
                for v in values:
                    wr_stats_line += "%0.5f, " % v
            # 2) 最后写各 accelerator 在该 component 上的平均 ratio
            for accelerator in accelerator_list:
                mean_val = energy_sum[comp][accelerator] / num_models
                wr_stats_line += "%0.5f, " % mean_val
            wr_stats_line += "\n"
            ff.write(wr_stats_line)


def main():
    global batch_size, accelerator_list, model_name_dict, normalized_bench
    parser = argparse.ArgumentParser(description="Run BitFusion simulations for mixed-precision accelerators.")
    parser.add_argument(
        "--models",
        type=str,
        default="llama3_8b",
        help="Comma-separated list of model names to simulate "
             "(must match names used in accel_model_configs.MODELS / benchmarks.benchlist)."
    )
    parser.add_argument(
        "--accelerators",
        type=str,
        default="olive,ant,mant,microscopiq,m2xfp,nvesm2",
        help="Comma-separated list of accelerator schemes (e.g., 'olive,ant,mant,microscopiq,m2xfp,nvesm2')."
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size used in the simulation."
    )
    parser.add_argument(
        "--normalized-bench",
        type=str,
        default=None,
        help=(
            "Accelerator scheme used as baseline for normalization. "
            "If not specified, defaults to the first entry in --accelerators."
        ),
    )

    args = parser.parse_args()

    # 1) Parse models and bench types
    model_list = [m.strip() for m in args.models.split(",") if m.strip()]
    accelerator_list = [b.strip() for b in args.accelerators.split(",") if b.strip()]

    if not accelerator_list:
        raise ValueError("accelerators list is empty. Please pass --accelerators=olive,ant,...")

    # 2) Decide normalized_bench
    normalized_bench_arg = (args.normalized_bench or "").strip()
    if normalized_bench_arg:
        normalized_bench = normalized_bench_arg
        if normalized_bench not in accelerator_list:
            raise ValueError(
                f"normalized_bench='{normalized_bench}' must be one of accelerators={accelerator_list}"
            )
    else:
        # default: first accelerator in accelerator_list
        normalized_bench = accelerator_list[0]

    # 2) Override benchmarks.benchlist so check_pandas_or_run / get_bench_nn 使用统一的模型列表

    benchmarks.benchlist = model_list

    # 3) Update batch size
    batch_size = args.batch_size

    # 4) Build model_name_dict
    model_name_dict = {name: name for name in model_list}

    # Run simulations
    for accelerator in accelerator_list:
        run_sim(accelerator)

    # Post-process results
    process_result()

if __name__ == "__main__":
    main()
