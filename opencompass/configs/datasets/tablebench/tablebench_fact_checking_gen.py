"""TableBench Fact Checking任务配置
qtype='FactChecking' 的各种子任务
"""
from mmengine.config import read_base

with read_base():
    from .tablebench_gen_base import (
        TABLEBENCH_HF_PATH,
        tablebench_base_reader_cfg,
        TableBenchDataset,
        TableBenchEvaluator,
        PromptTemplate,
        ZeroRetriever,
        GenInferencer,
    )

# ===== Fact Checking =====
tablebench_fact_reader_cfg = tablebench_base_reader_cfg.copy()

tablebench_fact_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(
            round=[
                dict(
                    role='HUMAN',
                    prompt="""Given the table below, verify the statement.

Table:
{table}

Statement: {question}

Please verify if this statement is correct based on the table.

Answer:"""
                ),
            ],
        ),
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer, max_out_len=256),
)

tablebench_fact_eval_cfg = dict(
    evaluator=dict(type=TableBenchEvaluator, metric='exact_match')
)

# ===== Dataset Definitions =====
tablebench_fact_checking_datasets = []

# FactChecking 类型的任务
tablebench_fact_checking_datasets.append(
    dict(
        abbr='tablebench_fact_checking',
        type=TableBenchDataset,
        path=TABLEBENCH_HF_PATH,
        qtype='FactChecking',
        reader_cfg=tablebench_fact_reader_cfg,
        infer_cfg=tablebench_fact_infer_cfg,
        eval_cfg=tablebench_fact_eval_cfg,
    )
)