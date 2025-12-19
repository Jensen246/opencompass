"""TableBench Numerical Reasoning任务配置
qtype='NumericalReasoning' 的各种子任务
"""
from mmengine.config import read_base

with read_base():
    from .tablebench_gen_base import (
        TABLEBENCH_HF_PATH,
        tablebench_base_reader_cfg,
        TableBenchDataset,
        TableBenchNumericEvaluator,
        PromptTemplate,
        ZeroRetriever,
        GenInferencer,
    )

# ===== Numerical Reasoning =====
tablebench_numerical_reader_cfg = tablebench_base_reader_cfg.copy()

tablebench_numerical_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(
            round=[
                dict(
                    role='HUMAN',
                    prompt="""Based on the table below, perform the numerical reasoning task.

Table:
{table}

Task: {question}

Please provide your answer as a number.

Answer:"""
                ),
            ],
        ),
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer, max_out_len=512),
)

tablebench_numerical_eval_cfg = dict(
    evaluator=dict(type=TableBenchNumericEvaluator, tolerance=1e-2)
)

# ===== Dataset Definitions =====
tablebench_numerical_datasets = []

# NumericalReasoning 类型的任务（不指定 qsubtype，加载所有）
tablebench_numerical_datasets.append(
    dict(
        abbr='tablebench_numerical',
        type=TableBenchDataset,
        path=TABLEBENCH_HF_PATH,
        qtype='NumericalReasoning',
        reader_cfg=tablebench_numerical_reader_cfg,
        infer_cfg=tablebench_numerical_infer_cfg,
        eval_cfg=tablebench_numerical_eval_cfg,
    )
)