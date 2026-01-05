"""TableBench基础配置 - 通用设置"""
from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets.tablebench import TableBenchDataset, TableBenchEvaluator, TableBenchNumericEvaluator

# HuggingFace dataset path
TABLEBENCH_HF_PATH = 'Multilingual-Multimodal-NLP/TableBench'

# 通用的reader配置
tablebench_base_reader_cfg = dict(
    input_columns=['table', 'question'],
    output_column='answer'
)

# 导出基础配置供其他文件使用
__all__ = [
    'TABLEBENCH_HF_PATH',
    'tablebench_base_reader_cfg',
    'TableBenchDataset',
    'TableBenchEvaluator',
    'TableBenchNumericEvaluator',
    'PromptTemplate',
    'ZeroRetriever',
    'GenInferencer',
]