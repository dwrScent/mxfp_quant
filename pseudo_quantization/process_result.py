# from fileinput import filename 
# from math import ceil
from os import error
# from types import resolve_bases
import re
import os
import sys
# from nbformat import read
import numpy as np
from numpy.core.defchararray import startswith
from numpy.lib.function_base import append
import argparse
import subprocess

# 全局变量，表头只写入一次
flint_ext_trigger = False
ant_trigger = False

def get_files():
    model_files=[]
    for root, dirs, files in os.walk(os.getcwd()):
        for filename in files:
            # 拼接文件的完整路径
            full_path = os.path.join(root, filename)
            sub_filename = full_path.split('/', -1)[-1]
            # if sub_filename.endswith('.out'):
            if sub_filename.endswith('.log'):
            # 处理文件，可以在这里调用函数等进行处理
                print(full_path)
                model_files.append(full_path)
    return model_files

def read_metrics(filename):
    global flint_ext_trigger
    output_file = f'stats.txt'

    acc = -1
    with open(filename, 'r') as f_in, open(output_file, 'a') as f_out:
        # 逐行读取文件内容
        # 每个模型的 metrics
        task_name = r"\|(\w+)\s+\|"
        # include_metrics = [r"word_perplexity\|(\d+\.\d+)", r"acc\s+\|(\d+\.\d+)", r"ppl:  (\d+\.\d+)", r"overall_mse: (\d+\.\d+)", r"mc2\s+\|(\d+\.\d+)"]
        include_metrics = [r"word_perplexity\|(\d+\.\d+)", r"acc\s+\|(\d+\.\d+)", r"ppl:  (\d+\.\d+)", r"mc2\s+\|(\d+\.\d+)"]
        task_dict = {}
        for line in f_in:
            for pattern in include_metrics:
                match = re.search(pattern, line)
                if match:
                    if pattern.startswith('acc'):
                        name_match = re.search(task_name, line)
                        acc = float(match.group(1))
                        task_dict[name_match.group(1)] = acc
                    elif pattern.startswith('ppl'):
                        name_match = "wikitext"
                        acc = float(match.group(1))
                        task_dict[name_match] = acc
                    else:
                        name_match = "values"
                        acc = float(match.group(1))
                        task_dict[name_match] = acc
                        

        # filename 是绝对路径，写入时只需要文件的名字
        output_name = filename.split('/')[-1]
        # output_name = output_name.split('.')[0]
        wr_line = f'{output_name}, '
        header_line = "file, "
        for task_name in task_dict:
            header_line += f'{task_name}, '
            wr_line += f'{task_dict[task_name]}, '
        
        # 以 flint_0, flint_1... 的顺序写入

        wr_line += '\n'
        header_line += '\n'
        if not flint_ext_trigger:
            f_out.write(header_line)
            flint_ext_trigger = True
        f_out.write(wr_line)

def recompile():
    files = get_files()
    if not files:
        print("no files")
        return
    for file in files:
        read_metrics(file)
    

if __name__ == '__main__':
    # parser = argparse.ArgumentParser(description='Modify control code in SASS assembly files.')
    # parser.add_argument('--mode', choices=['gto', 'lrr'], help='the mode of control code modification', required=True)
    # parser.add_argument('--method', choices=['alu', 'ctrl', 'no_setp'], default='ctrl', help='the mode of control code modification')
    # args = parser.parse_args()

    # mode = args.mode
    # method = args.method
    recompile()
