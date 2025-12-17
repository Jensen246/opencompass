from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets import TableBenchDataset, TableBenchEvaluator

# Reader configuration
tablebench_reader_cfg = dict(
    input_columns=['table', 'question'],
    output_column='answer'
)

# Chain-of-Thought prompting for TableBench
tablebench_cot_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(
            round=[
                dict(
                    role='HUMAN',
                    prompt="""Given the table below, answer the question step by step.

{table}

Question: {question}

Let's break this down:
1. First, identify the relevant information in the table
2. Then, apply the necessary reasoning
3. Finally, provide the answer

Step-by-step reasoning:"""
                ),
            ],
        ),
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer, max_out_len=1024),
)

tablebench_cot_eval_cfg = dict(
    evaluator=dict(type=TableBenchEvaluator, metric='exact_match')
)

# Dataset definitions with CoT
tablebench_cot_datasets = []

# Use official TableBench task names
task_types = [
    ('TQA', 'tqa'),
    ('TFV', 'tfv'),
    ('TCR', 'tcr'),
]

for task_name, abbr_suffix in task_types:
    for split in ['dev', 'test']:
        tablebench_cot_datasets.append(
            dict(
                abbr=f'tablebench_{abbr_suffix}_{split}_cot',
                type=TableBenchDataset,
                path='./data/tablebench',
                task_type=f'{task_name}_{split}',
                reader_cfg=tablebench_reader_cfg,
                infer_cfg=tablebench_cot_infer_cfg,
                eval_cfg=tablebench_cot_eval_cfg,
            )
        )