"""TableBench Data Analysis任务配置
qtype='DataAnalysis' 的各种子任务
"""
from mmengine.config import read_base

with read_base():
    from .tablebench_gen_base import (
        TABLEBENCH_HF_PATH,
        tablebench_base_reader_cfg,
        TableBenchDataset,
        TableBenchEvaluator,
        TableBenchNumericEvaluator,
        PromptTemplate,
        ZeroRetriever,
        GenInferencer,
    )

# ===== Statistical Analysis =====
tablebench_statistical_reader_cfg = tablebench_base_reader_cfg.copy()

tablebench_statistical_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(
            round=[
                dict(
                    role='HUMAN',
                    prompt="""Analyze the table below and answer the question.

Table:
{table}

Question: {question}

Please provide a concise and accurate answer based on the table data.

Answer:"""
                ),
            ],
        ),
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer, max_out_len=512),
)

tablebench_statistical_eval_cfg = dict(
    evaluator=dict(type=TableBenchNumericEvaluator, tolerance=1e-2)
)

# ===== Data Analysis (general) =====
tablebench_data_analysis_general_reader_cfg = tablebench_base_reader_cfg.copy()

tablebench_data_analysis_general_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(
            round=[
                dict(
                    role='HUMAN',
                    prompt="""Analyze the table below and answer the question.

Table:
{table}

Question: {question}

Answer:"""
                ),
            ],
        ),
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer, max_out_len=512),
)

tablebench_data_analysis_general_eval_cfg = dict(
    evaluator=dict(type=TableBenchEvaluator, metric='exact_match')
)

# ===== Dataset Definitions =====
tablebench_data_analysis_datasets = []

# 定义 DataAnalysis 类型的各种子任务
# 格式: (qsubtype, abbr_suffix, reader_cfg, infer_cfg, eval_cfg)
data_analysis_tasks = [
    ('StatisticalAnalysis', 'stat', tablebench_statistical_reader_cfg, tablebench_statistical_infer_cfg, tablebench_statistical_eval_cfg),
    # 如果没有 qsubtype 过滤，加载所有 DataAnalysis 任务
    (None, 'all', tablebench_data_analysis_general_reader_cfg, tablebench_data_analysis_general_infer_cfg, tablebench_data_analysis_general_eval_cfg),
]

for qsubtype, abbr_suffix, reader_cfg, infer_cfg, eval_cfg in data_analysis_tasks:
    dataset_cfg = dict(
        abbr=f'tablebench_analysis_{abbr_suffix}',
        type=TableBenchDataset,
        path=TABLEBENCH_HF_PATH,
        qtype='DataAnalysis',
        reader_cfg=reader_cfg,
        infer_cfg=infer_cfg,
        eval_cfg=eval_cfg,
    )
    
    # 只有当 qsubtype 不是 None 时才添加
    if qsubtype:
        dataset_cfg['qsubtype'] = qsubtype
    
    tablebench_data_analysis_datasets.append(dataset_cfg)