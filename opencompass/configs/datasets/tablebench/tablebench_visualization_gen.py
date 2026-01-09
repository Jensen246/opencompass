"""TableBench Visualization任务配置
qtype='Visualization' 的各种子任务
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

# ===== Visualization =====
tablebench_viz_reader_cfg = tablebench_base_reader_cfg.copy()

tablebench_viz_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(
            round=[
                dict(
                    role='HUMAN',
                    prompt="""Based on the table below, answer the visualization question.

Table:
{table}

Question: {question}

Answer:"""
                ),
            ],
        ),
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer, max_out_len=16384),
)

tablebench_viz_eval_cfg = dict(
    evaluator=dict(type=TableBenchEvaluator, metric='f1')
)

# ===== Dataset Definitions =====
tablebench_visualization_datasets = []

# Visualization 类型的任务
tablebench_visualization_datasets.append(
    dict(
        abbr='tablebench_visualization',
        type=TableBenchDataset,
        path=TABLEBENCH_HF_PATH,
        qtype='Visualization',
        instruction_type='SCoT',
        reader_cfg=tablebench_viz_reader_cfg,
        infer_cfg=tablebench_viz_infer_cfg,
        eval_cfg=tablebench_viz_eval_cfg,
    )
)