from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets import TableBenchDataset, TableBenchEvaluator

# Few-shot examples adapted for TableBench format
few_shot_prompt = """Here are some examples:

Example 1:
Table:
| Name  | Age | City |
|-------|-----|------|
| Alice | 28  | NYC  |
| Bob   | 35  | LA   |

Question: What is Bob's age?
Answer: 35

Example 2:
Table:
| Product | Price | Stock |
|---------|-------|-------|
| Apple   | 2.50  | 100   |
| Banana  | 1.20  | 150   |

Question: How much does a banana cost?
Answer: 1.20

Now answer this question:

{table}

Question: {question}

Answer:"""

tablebench_reader_cfg = dict(
    input_columns=['table', 'question'],
    output_column='answer'
)

tablebench_few_shot_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(
            round=[
                dict(
                    role='HUMAN',
                    prompt=few_shot_prompt
                ),
            ],
        ),
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer, max_out_len=512),
)

tablebench_few_shot_eval_cfg = dict(
    evaluator=dict(type=TableBenchEvaluator, metric='exact_match')
)

tablebench_few_shot_datasets = []

# Use official task names
for task_name, abbr in [('TQA', 'tqa'), ('TFV', 'tfv')]:
    for split in ['dev', 'test']:
        tablebench_few_shot_datasets.append(
            dict(
                abbr=f'tablebench_{abbr}_{split}_few_shot',
                type=TableBenchDataset,
                path='./data/tablebench',
                task_type=f'{task_name}_{split}',
                reader_cfg=tablebench_reader_cfg,
                infer_cfg=tablebench_few_shot_infer_cfg,
                eval_cfg=tablebench_few_shot_eval_cfg,
            )
        )