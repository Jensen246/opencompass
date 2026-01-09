"""TableBench Numerical Reasoning任务配置
qtype='NumericalReasoning' 的各种子任务
"""
from mmengine.config import read_base

with read_base():
    from .tablebench_gen_base import (
        TABLEBENCH_HF_PATH,
        tablebench_base_reader_cfg,
        TableBenchDataset,
        PromptTemplate,
        ZeroRetriever,
        GenInferencer,
    )

from opencompass.datasets import CustomDataset, generic_llmjudge_postprocess
from opencompass.evaluator import GenericLLMEvaluator

GRADER_TEMPLATE_NUMERICAL = """
Please as a grading expert, judge whether the numerical answer given by the candidate is correct compared to the standard answer.

Here are some evaluation criteria:
1. The standard answer is always correct. You only need to judge whether the candidate's answer matches the standard answer.
2. For numerical answers, consider answers correct if they are mathematically equivalent or within a very small tolerance (±0.01).
3. Ignore formatting differences:
   - "1000" = "1,000" = "1000.0"
   - "0.5" = "50%" = "1/2"
   - "$100" = "100 dollars"
4. If the prediction contains "Final Answer:", extract the numerical value after this marker.
5. If the candidate provides a calculation process, only judge the final numerical result.
6. Consider scientific notation and different units if applicable (but they should represent the same value).
7. If the candidate's answer is invalid (e.g., incomplete, non-numerical where a number is expected, or states it cannot answer), select option C (INVALID).

Please judge whether the following answers are consistent with the standard answer based on the above criteria. Grade the predicted answer as one of:
A: CORRECT 
B: INCORRECT
C: INVALID

Just return the letters "A", "B", or "C", with no text around it.

Here is your task. Simply reply with either CORRECT, INCORRECT, or INVALID.

<Table Context>:
{table}
<Question>:
{question}
<Standard Answer>:
{answer}
<Candidate's Answer>:
{prediction}

Judging the correctness of the candidate's answer:
""".strip()


# ===== Numerical Reasoning =====
tablebench_numerical_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(
            round=[
                dict(
                    role='HUMAN',
                    prompt="""{instruction}

Table:
{table}

Question: {question}

Please analyze the table and provide the numerical answer. End your response with "Final Answer: <your numerical answer>".

Answer:"""
                ),
            ],
        ),
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer, max_out_len=512),
)

tablebench_numerical_eval_cfg = dict(
    evaluator=dict(
        type=GenericLLMEvaluator,
        prompt_template=dict(
            type=PromptTemplate,
            template=dict(
                begin=[
                    dict(
                        role='SYSTEM',
                        fallback_role='HUMAN',
                        prompt="You are a helpful assistant who evaluates the correctness of numerical answers for table reasoning tasks.",
                    )
                ],
                round=[
                    dict(role='HUMAN', prompt=GRADER_TEMPLATE_NUMERICAL),
                ],
            ),
        ),
        dataset_cfg=dict(
            type=TableBenchDataset,
            path=TABLEBENCH_HF_PATH,
            qtype='NumericalReasoning',
            instruction_type='SCoT',
            reader_cfg=tablebench_base_reader_cfg,
        ),
        judge_cfg=dict(),
        dict_postprocessor=dict(type=generic_llmjudge_postprocess),
    ),
)

# ===== Dataset Definitions =====
tablebench_numerical_datasets = [
    dict(
        abbr='tablebench_numerical',
        type=TableBenchDataset,
        path=TABLEBENCH_HF_PATH,
        qtype='NumericalReasoning',
        instruction_type='SCoT',
        reader_cfg=tablebench_base_reader_cfg,
        infer_cfg=tablebench_numerical_infer_cfg,
        eval_cfg=tablebench_numerical_eval_cfg,
    )
]

# # ===== Numerical Reasoning =====
# tablebench_numerical_reader_cfg = tablebench_base_reader_cfg.copy()

# tablebench_numerical_infer_cfg = dict(
#     prompt_template=dict(
#         type=PromptTemplate,
#         template=dict(
#             round=[
#                 dict(
#                     role='HUMAN',
#                     prompt="""Based on the table below, perform the numerical reasoning task.

# Table:
# {table}

# Task: {question}

# Please provide your answer as a number.

# Answer:"""
#                 ),
#             ],
#         ),
#     ),
#     retriever=dict(type=ZeroRetriever),
#     inferencer=dict(type=GenInferencer, max_out_len=512),
# )

# tablebench_numerical_eval_cfg = dict(
#     evaluator=dict(type=TableBenchNumericEvaluator, tolerance=1e-2)
# )

# # ===== Dataset Definitions =====
# tablebench_numerical_datasets = []

# # NumericalReasoning 类型的任务（不指定 qsubtype，加载所有）
# tablebench_numerical_datasets.append(
#     dict(
#         abbr='tablebench_numerical',
#         type=TableBenchDataset,
#         path=TABLEBENCH_HF_PATH,
#         qtype='NumericalReasoning',
#         reader_cfg=tablebench_numerical_reader_cfg,
#         infer_cfg=tablebench_numerical_infer_cfg,
#         eval_cfg=tablebench_numerical_eval_cfg,
#     )
# )