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
                    # 使用数据集自带的 instruction
                    prompt="""{instruction}

Table:
{table}

Question: {question}

Please analyze the table and provide your answer. End your response with "Final Answer: <your answer>" where the answer should be True/False or Yes/No.

Answer:"""
                ),
            ],
        ),
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer, max_out_len=256),
)

tablebench_fact_eval_cfg = dict(
    evaluator=dict(type=TableBenchEvaluator, metric='exact_match')  # 改用 f1，更宽松
)

# ===== Dataset Definitions =====
tablebench_fact_checking_datasets = []

tablebench_fact_checking_datasets.append(
    dict(
        abbr='tablebench_fact_checking',
        type=TableBenchDataset,
        path=TABLEBENCH_HF_PATH,
        qtype='FactChecking',
        instruction_type=None,  # 明确指定
        reader_cfg=tablebench_fact_reader_cfg,
        infer_cfg=tablebench_fact_infer_cfg,
        eval_cfg=tablebench_fact_eval_cfg,
    )
)