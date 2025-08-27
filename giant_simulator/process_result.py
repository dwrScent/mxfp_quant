import re
import ast
import numpy as np
import argparse # 1. 导入 argparse 库

def parse_data(log_content):
    """Parses the log text and returns a structured dictionary."""
    parsed_data = {}
    cycle_pattern = re.compile(r"(\w+) cycle (\[.*?\])")
    energy_pattern = re.compile(r"(\w+) energy (\[\[.*?\]\])")

    for method, cycles_str in cycle_pattern.findall(log_content):
        if method not in parsed_data:
            parsed_data[method] = {}
        parsed_data[method]['cycle'] = ast.literal_eval(cycles_str)

    for method, energy_str in energy_pattern.findall(log_content):
        if method not in parsed_data:
            parsed_data[method] = {}
        parsed_data[method]['energy'] = ast.literal_eval(energy_str)
    return parsed_data

def generate_raw_csv(parsed_data, models, methods, energy_components):
    """Generates the CSV string for the raw data."""
    metrics = ['Cycle'] + energy_components
    header1 = ['']
    header2 = ['Metrics']
    for model_name in models:
        header1.extend([model_name] + [''] * (len(methods) - 1))
        header2.extend(methods)
        
    csv_lines = [",".join(header1), ",".join(header2)]

    for metric_idx, metric_name in enumerate(metrics):
        row = [metric_name]
        for model_idx, model_name in enumerate(models):
            for method in methods:
                if metric_name == 'Cycle':
                    value = parsed_data[method]['cycle'][model_idx]
                else:
                    component_idx = energy_components.index(metric_name)
                    value = parsed_data[method]['energy'][model_idx][component_idx]
                row.append(f"{value:.2f}")
        csv_lines.append(",".join(row))
        
    return "\n".join(csv_lines)

def generate_normalized_csv(parsed_data, models, methods, energy_components, baseline_method='olive'):
    """Generates the CSV string for the normalized data."""
    metrics = ['Cycle'] + energy_components + ['Total Energy']
    header1 = ['']
    header2 = ['Metrics']
    for model_name in models:
        header1.extend([model_name] + [''] * (len(methods) - 1))
        header2.extend(methods)
        
    csv_lines = [",".join(header1), ",".join(header2)]

    for metric_name in metrics:
        row = [metric_name]
        for model_idx, model_name in enumerate(models):
            baseline_cycle = parsed_data[baseline_method]['cycle'][model_idx]
            total_baseline_energy = sum(parsed_data[baseline_method]['energy'][model_idx])

            for method in methods:
                if metric_name == 'Cycle':
                    raw_value = parsed_data[method]['cycle'][model_idx]
                    normalized_value = raw_value / baseline_cycle
                elif metric_name == 'Total Energy':
                    total_energy = sum(parsed_data[method]['energy'][model_idx])
                    normalized_value = total_energy / total_baseline_energy
                else:
                    component_idx = energy_components.index(metric_name)
                    raw_value = parsed_data[method]['energy'][model_idx][component_idx]
                    normalized_value = raw_value / total_baseline_energy
                
                row.append(f"{normalized_value:.4f}")
        csv_lines.append(",".join(row))
        
    return "\n".join(csv_lines)

# 2. 主函数入口
if __name__ == "__main__":
    # 使用 argparse 定义命令行参数
    parser = argparse.ArgumentParser(description="Parse simulator log file and generate CSV reports.")
    parser.add_argument("logfile", type=str, help="Path to the log file to be processed.")
    args = parser.parse_args()

    # 3. 从文件读取日志数据
    try:
        with open(args.logfile, 'r') as f:
            log_data = f.read()
    except FileNotFoundError:
        print(f"Error: Log file not found at '{args.logfile}'")
        exit(1)

    # 定义模型和方法顺序
    methods_in_log = ['olive', 'ant', 'microscopiq', 'm2xfp']
    # methods_in_log = ['olive', 'giant']
    models_in_log = ['opt6b7', 'llama2_7b', 'falcon_7b', 'llama3_8b', 'mistral_7b', 'llama3_70b']
    # models_in_log = ['llama3_70b']  # 仅处理 llama3_70b
    energy_comps = ['Static', 'Dram', 'Buffer', 'Core']

    # 1. 解析数据
    data = parse_data(log_data)

    # 2. 生成并打印原始数据表格
    print("--- Raw Data ---")
    raw_csv_output = generate_raw_csv(data, models_in_log, methods_in_log, energy_comps)
    print(raw_csv_output)
    print("\n" + "="*80 + "\n")

    # 3. 生成并打印归一化数据表格
    print("--- Normalized Data (Baseline: olive) ---")
    normalized_csv_output = generate_normalized_csv(data, models_in_log, methods_in_log, energy_comps, baseline_method='olive')
    print(normalized_csv_output)