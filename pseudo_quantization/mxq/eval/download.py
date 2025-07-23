import os
from huggingface_hub import snapshot_download, login

# 定义数据集信息
datasets_info = [
    {
        "Dataset": "AIME-90",
        "Local Dir": "./datasets/AIME90",
        "URL": "https://huggingface.co/datasets/xiaoyuanliu/AIME90"
    },
    {
        "Dataset": "AIME-2025",
        "Local Dir": "./datasets/aime_2025",
        "URL": "https://huggingface.co/datasets/yentinglin/aime_2025"
    },
    {
        "Dataset": "MATH-500",
        "Local Dir": "./datasets/MATH-500",
        "URL": "https://huggingface.co/datasets/HuggingFaceH4/MATH-500" 
    },
    {
        "Dataset": "GSM8K",
        "Local Dir": "./datasets/gsm8k",
        "URL": "https://huggingface.co/datasets/openai/gsm8k"
    },
    {
        "Dataset": "GPQA-Diamond",
        "Local Dir": "./datasets/gpqa",
        "URL": "https://huggingface.co/datasets/Idavidrein/gpqa"
    },
    {
        "Dataset": "LiveCodeBench",
        "Local Dir": "./datasets/code_generation_lite",
        "URL": "https://huggingface.co/datasets/livecodebench/code_generation_lite"
    }
]

# 设置基础目录（如果你想把所有数据集放在一个根目录下）
# 例如，如果你的脚本在项目根目录，想把所有数据集放到 './data' 目录下
# base_output_dir = "./data" 
# 这里我们直接使用表格中的相对路径，所以不需要 base_output_dir

def download_dataset_to_local_path(repo_id: str, local_dir: str):
    print(f"正在处理数据集：{repo_id}")
    print(f"目标本地目录：{local_dir}")
    os.makedirs(local_dir,  exist_ok=True)
 
    try:
        snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            local_dir=local_dir,
            local_dir_use_symlinks=False,
        )
        print(f"成功下载 {repo_id} 到 {local_dir}\n")
    except Exception as e:
        print(f"下载 {repo_id} 失败: {e}\n")

def main():
    print("开始下载数据集...")

    for dataset in datasets_info:
        # 从 URL 中提取 repo_id (例如 'xiaoyuanliu/AIME90')
        # URL 格式是 https://huggingface.co/datasets/<repo_id>
        url_parts = dataset["URL"].split('/datasets/')
        if len(url_parts) > 1:
            repo_id = url_parts[1]
            # 如果 URL 包含多余的 '/' (例如 'HuggingFaceH4/MATH-500/'), split again
            # 但对于 datasets 命名规则，通常只有 repo_id
            
            # 使用 split('/') 再次处理，确保得到正确的 repo_id，例如 "xiaoyuanliu/AIME90"
            # 注意：对于像 'HuggingFaceH4/MATH-500' 这样的，repo_id 就是 'HuggingFaceH4/MATH-500'
            # 所以直接取 parts[1] 是正确的。
            
            # 兼容处理可能不完整的 URL, 例如 'MATH-5C' 和 'code_gene'
            # 这里假定完整的 URL 应该是 HuggingFaceH4/MATH-500 和 livecodebench/code_generation_lite
            # 如果原始URL是 'https://huggingface.co/datasets/HuggingFaceH4/MATH-5C'，那么repo_id就是 'HuggingFaceH4/MATH-5C'
            # 我们直接使用解析到的repo_id
            
            download_dataset_to_local_path(repo_id=repo_id, local_dir=dataset["Local Dir"])
        else:
            print(f"无法从 URL 获取 repo_id: {dataset['URL']}")
            print("请检查 URL 格式是否正确，它应该包含 '/datasets/'。")

    print("所有数据集处理完成。")

if __name__ == "__main__":
    main()