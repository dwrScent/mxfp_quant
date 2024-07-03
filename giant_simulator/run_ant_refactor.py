from re import I
import pandas
import configparser
import os
import numpy as np
import bitfusion.src.benchmarks.benchmarks as benchmarks
from bitfusion.src.simulator.stats import Stats
from bitfusion.src.simulator.simulator import Simulator
from bitfusion.src.sweep.sweep import SimulatorSweep, check_pandas_or_run
from bitfusion.src.utils.utils import *
from bitfusion.src.optimizer.optimizer import optimize_for_order, get_stats_fast
import copy

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
batch_size = 1

# directory to store the .csv
results_dir = './results'
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

bf_e_cycles = {}
bf_e_energy = {}

def run_sim(bench_type):
    # Get the configuration file for the given benchmark type
    config_file = f'conf_{bench_type}.ini'

    # Create simulator object
    bf_e_sim = Simulator(config_file, False)
    bf_e_sim_sweep_csv = os.path.join(results_dir, f'{bench_type}.csv')
    bf_e_sim_sweep_df = pandas.DataFrame(columns=sim_sweep_columns)
    bf_e_results = check_pandas_or_run(bf_e_sim, bf_e_sim_sweep_df, bf_e_sim_sweep_csv, batch_size=batch_size, bench_type=bench_type)
    bf_e_results = bf_e_results.groupby('Network',as_index=False).agg(np.sum)

    # Store the total cycles and energy for each network
    bf_e_cycles[bench_type] = []
    bf_e_energy[bench_type] = []
    for name in benchmarks.benchlist:
        bf_e_stats = df_to_stats(bf_e_results.loc[bf_e_results['Network'] == name])
        bf_e_cycles[bench_type].append(bf_e_stats.total_cycles)
        bf_e_energy[bench_type].append(bf_e_stats.get_energy_breakdown(bf_e_sim.get_energy_cost()))
    
    # Print the cycle and energy results for this benchmark type
    print(f"{bench_type} cycle", bf_e_cycles[bench_type])
    print(f"{bench_type} energy", bf_e_energy[bench_type])

model_name_dict = {
                #     'vgg16':'VGG16', 
                #    'resnet18':'ResNet18',
                #    'resnet50':'ResNet50',
                #    'inceptionv3':'InceptionV3',
                #    'vit':'ViT',
                #    'mnli':'BERT-MNLI',
                #    'cola':'BERT-CoLA',
                #    'sst_2':'BERT-SST-2',
                #     'bart_base':'bart_base',
                #     'bert_base':'bert_base',
                #     'bert_large':'bert_large',
                    # 'gpt2_xl':'gpt2_xl',
                    # 'bloom3b':'bloom3b',
                    # 'bloom7b1':'bloom7b1',
                    # 'opt6b7':'opt6b7',
                    'llama_7b':'llama_7b',
                    }
def process_result():
    with open(os.path.join(os.getcwd(), 'results', 'olive_res.csv'), "a") as ff:
        wr_stats_line = "Time, "
        wr_bench_name = ", "
        wr_model_name = ", "
        normalized_bench = 'bit'

        bf_e_cycles_length = len(bf_e_cycles[normalized_bench])
        tmp_cycle = {}
        tmp_cycle_mean = {}

        # 初始化 geomean cycle
        for bench_type in bf_e_cycles:
            tmp_cycle_mean[bench_type] = 0
        all_cyc = []

        normalized_cycle = bf_e_cycles[normalized_bench]
        for i in range(bf_e_cycles_length):
            model_name = benchmarks.benchlist[i]
            for bench_type, cycles in bf_e_cycles.items():
                tmp_cycle[bench_type] = cycles[i] / normalized_cycle[i]
                tmp_cycle_mean[bench_type] += tmp_cycle[bench_type]

                all_cyc.append(tmp_cycle[bench_type])
                wr_bench_name += f"{bench_type}, "
                wr_stats_line += "%0.5f, " %(tmp_cycle[bench_type])
            wr_model_name += f"{model_name_dict[model_name]}, , , , , , "

        # 处理并写入 Geomean 的数据
        for bench_type, cycles in bf_e_cycles.items():
            tmp_cycle_mean[bench_type] /= bf_e_cycles_length
            wr_bench_name += f"{bench_type}, "
            wr_stats_line += "%0.5f, " %(tmp_cycle_mean[bench_type])
        
        wr_model_name += "Geomean, , , , , \n"
        wr_bench_name += "\n"
        ff.write(wr_model_name)
        ff.write(wr_bench_name)
        ff.write(wr_stats_line)

        # process energy
        wr_stats_line = ""
        all_energy = {}
        tmp_energy = {}
        tmp_energy_total = {}
        tmp_energy_mean = {}
        energy_list = ['Static', 'Dram', 'Buffer', 'Core']
        for i in range(bf_e_cycles_length):
            model_name = benchmarks.benchlist[i]
            tmp_energy_total[normalized_bench] = 0
            for item in bf_e_energy[normalized_bench][i]:
                tmp_energy_total[normalized_bench] += item
            normalized_energy = tmp_energy_total[normalized_bench]
            for bench_type, energy in bf_e_energy.items():
                tmp_energy[bench_type] = []
                tmp_energy_mean[bench_type] = 0
                for item in energy[i]:
                    tmp_energy[bench_type].append(item / normalized_energy)
            
            for index, item in enumerate(energy_list):
                all_energy[item] = {}
                all_energy[item][model_name] = {}
                tmp_list = []
                for bench_type, energy in tmp_energy.items():
                    tmp_list.append(energy[index])
                all_energy[item][model_name] = tmp_list

        ff.write(wr_model_name)
        ff.write(wr_bench_name)
        for index, item in enumerate(energy_list):
            wr_stats_line = f"{item}, "
            for value in all_energy[item].values():
                for idx, bench_type in enumerate(bench_type_list):
                    wr_stats_line += "%0.5f, " %(value[idx])
                    tmp_energy_mean[bench_type] += value[idx]
            for idx, bench_type in enumerate(bench_type_list):
                wr_stats_line += "%0.5f, " %(np.mean(tmp_energy_mean[bench_type]))
            wr_stats_line += "\n"
            ff.write(wr_stats_line)


# bench_type_list = ['olive', 'ant', 'ola', 'ada']
bench_type_list = ['giant', 'olive', 'ant', 'bit']
# bench_type_list = ['bit']

for item in bench_type_list:
    run_sim(item)

process_result()