#!/bin/bash
set -e

# 获取脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# 激活conda环境
source ~/miniconda3/bin/activate opencompass

# 设置环境变量
export OPENAI_API_KEY="sk-1234"
export OPENAI_BASE_URL="http://localhost:3000/v1/"

# LLM Judge 环境变量 (用于TableBench等需要LLM评估的任务)
export OC_JUDGE_MODEL="gpt-5.1"
export OC_JUDGE_API_KEY="sk-1234"
export OC_JUDGE_API_BASE="http://localhost:3000/v1/"

# 进入项目根目录
cd "$PROJECT_ROOT"

# 生成配置文件
echo "Generating config files..."
python "$SCRIPT_DIR/generate_configs.py"

# 定义所有benchmark
BENCHMARKS=(
    "panorama/par4pc_cot"
    "panorama/pi4pc_cot"
    "panorama/noc4pc_cot"
    # "chemcotbench/mol_und"
    # "chemcotbench/mol_edit"
    # "chemcotbench/mol_opt"
    # "chemcotbench/reaction"
    # "tablebench/data_analysis"
    # "tablebench/fact_checking"
    # "tablebench/numerical_reasoning"
    # "tablebench/visualization"
    # "bioprobench/gen"
    # "bioprobench/ord"
    # "bioprobench/err"
    # "bioprobench/pqa"
)

# 并发运行所有benchmark
echo "Starting parallel benchmark tests..."

for benchmark in "${BENCHMARKS[@]}"; do
    config_path="$SCRIPT_DIR/$benchmark/config.py"
    echo "Running: $benchmark"
    python run.py "$config_path" &
done

# 等待所有后台任务完成
wait

echo "All benchmark tests completed!"
