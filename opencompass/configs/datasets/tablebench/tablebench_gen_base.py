from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets import TableBenchDataset, TableBenchEvaluator, TableBenchNumericEvaluator

# ===== Table Question Answering (TQA) =====
tablebench_tqa_reader_cfg = dict(
    input_columns=['table', 'question'],
    output_column='answer'
)

tablebench_tqa_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(
            round=[
                dict(
                    role='HUMAN',
                    prompt="""Based on the table below, answer the following question.

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

tablebench_tqa_eval_cfg = dict(
    evaluator=dict(type=TableBenchEvaluator, metric='exact_match')
)

# ===== Table Fact Verification (TFV) =====
tablebench_tfv_reader_cfg = dict(
    input_columns=['table', 'question'],
    output_column='answer'
)

tablebench_tfv_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(
            round=[
                dict(
                    role='HUMAN',
                    prompt="""Given the table below, determine if the following statement is true or false.

{table}

Statement: {question}

Answer (True/False):"""
                ),
            ],
        ),
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer, max_out_len=128),
)

tablebench_tfv_eval_cfg = dict(
    evaluator=dict(type=TableBenchEvaluator, metric='exact_match')
)

# ===== Table Cell Reasoning (TCR) =====
tablebench_tcr_reader_cfg = dict(
    input_columns=['table', 'question'],
    output_column='answer'
)

tablebench_tcr_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(
            round=[
                dict(
                    role='HUMAN',
                    prompt="""Based on the table, perform the reasoning task.

{table}

Task: {question}

Answer:"""
                ),
            ],
        ),
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer, max_out_len=512),
)

tablebench_tcr_eval_cfg = dict(
    evaluator=dict(type=TableBenchNumericEvaluator, tolerance=1e-2)
)

# ===== Dataset Definitions =====
tablebench_datasets = []

# Official TableBench tasks
# Note: Adjust task names based on your actual data files
tablebench_tasks = [
    ('TQA', 'tqa', tablebench_tqa_reader_cfg, tablebench_tqa_infer_cfg, tablebench_tqa_eval_cfg),
    ('TFV', 'tfv', tablebench_tfv_reader_cfg, tablebench_tfv_infer_cfg, tablebench_tfv_eval_cfg),
    ('TCR', 'tcr', tablebench_tcr_reader_cfg, tablebench_tcr_infer_cfg, tablebench_tcr_eval_cfg),
]

for task_name, abbr_suffix, reader_cfg, infer_cfg, eval_cfg in tablebench_tasks:
    for split in ['dev', 'test']:
        tablebench_datasets.append(
            dict(
                abbr=f'tablebench_{abbr_suffix}_{split}',
                type=TableBenchDataset,
                path='./data/tablebench',
                task_type=f'{task_name}_{split}',
                reader_cfg=reader_cfg,
                infer_cfg=infer_cfg,
                eval_cfg=eval_cfg,
            )
        )