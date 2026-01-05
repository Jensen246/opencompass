"""TableBench Fact Checking任务配置
qtype='FactChecking' 的各种子任务
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

GRADER_TEMPLATE_FACT_CHECKING = """
Please as a grading expert, judge whether the candidate's fact-checking answer is correct compared to the standard answer.

Here are some evaluation criteria:
1. The standard answer is always correct. You only need to judge whether the candidate's answer matches the standard answer.
2. Fact-checking answers are typically binary: True/False, Yes/No, Correct/Incorrect, Supported/Not Supported, etc.
3. Consider semantic equivalence: "True" = "Yes" = "Correct" = "Supported", and "False" = "No" = "Incorrect" = "Not Supported".
4. If the prediction contains "Final Answer:", extract the answer after this marker.
5. Ignore case differences and minor formatting.
6. If the candidate provides reasoning, only judge the final conclusion, not the reasoning process.
7. If the candidate's answer is invalid (e.g., incomplete, irrelevant, ambiguous, or states it cannot answer), select option C (INVALID).

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

# ===== Fact Checking =====
tablebench_fact_infer_cfg = dict(
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
    evaluator=dict(
        type=GenericLLMEvaluator,
        prompt_template=dict(
            type=PromptTemplate,
            template=dict(
                begin=[
                    dict(
                        role='SYSTEM',
                        fallback_role='HUMAN',
                        prompt="You are a helpful assistant who evaluates the correctness of fact-checking outputs for table understanding tasks.",
                    )
                ],
                round=[
                    dict(role='HUMAN', prompt=GRADER_TEMPLATE_FACT_CHECKING),
                ],
            ),
        ),
        dataset_cfg=dict(
            type=TableBenchDataset,
            path=TABLEBENCH_HF_PATH,
            qtype='FactChecking',  # ✅ 添加过滤条件
            instruction_type='DP',  # ✅ 添加过滤条件
            reader_cfg=tablebench_base_reader_cfg,
        ),
        judge_cfg=dict(),
        dict_postprocessor=dict(type=generic_llmjudge_postprocess),
    ),
)

# ===== Dataset Definitions =====
tablebench_fact_checking_datasets = [
    dict(
        abbr='tablebench_fact_checking',
        type=TableBenchDataset,
        path=TABLEBENCH_HF_PATH,
        qtype='FactChecking',
        instruction_type='DP',
        reader_cfg=tablebench_base_reader_cfg,
        infer_cfg=tablebench_fact_infer_cfg,
        eval_cfg=tablebench_fact_eval_cfg,
    )
]

# ===== Fact Checking =====
# tablebench_fact_reader_cfg = tablebench_base_reader_cfg.copy()

# tablebench_fact_infer_cfg = dict(
#     prompt_template=dict(
#         type=PromptTemplate,
#         template=dict(
#             round=[
#                 dict(
#                     role='HUMAN',
#                     prompt="""Given the table below, verify the statement.

# Table:
# {table}

# Statement: {question}

# Please verify if this statement is correct based on the table.

# Answer:"""
#                 ),
#             ],
#         ),
#     ),
#     retriever=dict(type=ZeroRetriever),
#     inferencer=dict(type=GenInferencer, max_out_len=256),
# )

# tablebench_fact_eval_cfg = dict(
#     evaluator=dict(type=TableBenchEvaluator, metric='exact_match')
# )

# # ===== Dataset Definitions =====
# tablebench_fact_checking_datasets = []

# # FactChecking 类型的任务
# tablebench_fact_checking_datasets.append(
#     dict(
#         abbr='tablebench_fact_checking',
#         type=TableBenchDataset,
#         path=TABLEBENCH_HF_PATH,
#         qtype='FactChecking',
#         reader_cfg=tablebench_fact_reader_cfg,
#         infer_cfg=tablebench_fact_infer_cfg,
#         eval_cfg=tablebench_fact_eval_cfg,
#     )
# )