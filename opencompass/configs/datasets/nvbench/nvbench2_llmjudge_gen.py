from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.evaluator import GenericLLMEvaluator
from opencompass.datasets import NvBench2Dataset
from opencompass.datasets import generic_llmjudge_postprocess

# 模型推理的 prompt
INFER_PROMPT = '''Given the following database table schema:
{table_schema}

Natural language query: {nl_query}

Please generate the visualization specification (including chart type, data encoding, filtering, and aggregation operations) to answer the query.'''

# LLM 评判的 prompt
JUDGE_PROMPT = """Please evaluate whether the model's visualization specification correctly addresses the user's natural language query.

<Table Schema>
{table_schema}
</Table Schema>

<Natural Language Query>
{nl_query}
</Natural Language Query>

<Reference Answer>
{gold_answer}
</Reference Answer>

<Model Output>
{prediction}
</Model Output>

Evaluation criteria:
1. Chart type correctness - Does the model choose the appropriate chart type?
2. Data encoding mappings - Are the axes, colors, and other visual encodings correct?
3. Filtering operations - Are the correct filters applied?
4. Aggregation operations - Are the correct aggregations used?

Grade the model's output as:
A: CORRECT - The output is semantically equivalent to the reference answer
B: INCORRECT - The output does not correctly address the query

Just return the letter "A" or "B", with no text around it."""

nvbench2_reader_cfg = dict(
    input_columns=['nl_query', 'table_schema'],
    output_column='gold_answer',
)

nvbench2_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(
            round=[
                dict(role='HUMAN', prompt=INFER_PROMPT),
            ]
        )
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer),
)

nvbench2_eval_cfg = dict(
    evaluator=dict(
        type=GenericLLMEvaluator,
        prompt_template=dict(
            type=PromptTemplate,
            template=dict(
                begin=[
                    dict(
                        role='SYSTEM',
                        fallback_role='HUMAN',
                        prompt='You are a helpful assistant who evaluates the correctness of visualization specifications.'
                    )
                ],
                round=[
                    dict(role='HUMAN', prompt=JUDGE_PROMPT),
                ]
            ),
        ),
        dataset_cfg=dict(
            type=NvBench2Dataset,
            path='TianqiLuo/nvBench2.0',
            split='test',
            reader_cfg=nvbench2_reader_cfg,
        ),
        judge_cfg=dict(),
        dict_postprocessor=dict(type=generic_llmjudge_postprocess),
    ),
    pred_role='BOT',
)

nvbench2_datasets = [
    dict(
        abbr='nvbench2',
        type=NvBench2Dataset,
        path='TianqiLuo/nvBench2.0',
        split='test',
        reader_cfg=nvbench2_reader_cfg,
        infer_cfg=nvbench2_infer_cfg,
        eval_cfg=nvbench2_eval_cfg,
    )
]
