# """TableBench Fact Checking任务配置
# qtype='FactChecking' 的各种子任务
# """
# from mmengine.config import read_base

# with read_base():
#     from .tablebench_gen_base import (
#         TABLEBENCH_HF_PATH,
#         tablebench_base_reader_cfg,
#         TableBenchDataset,
#         PromptTemplate,
#         ZeroRetriever,
#         GenInferencer,
#     )

# from opencompass.datasets import CustomDataset, generic_llmjudge_postprocess
# from opencompass.evaluator import GenericLLMEvaluator

# GRADER_TEMPLATE_FACT_CHECKING = """
# Please as a grading expert, judge whether the candidate's answer is correct compared to the standard answer.

# Evaluation criteria:
# 1. The standard answer is always correct. Judge whether the candidate's answer matches it.

# 2. For specific information questions (asking for numbers, dates, names, etc.):
#    - The candidate must provide the specific information
#    - Answers like "Yes" or "No" are INCORRECT if the standard answer is specific data
#    - The answer must match the standard answer exactly or be semantically equivalent

# 3. For verification questions (Yes/No, True/False):
#    - Accept equivalent expressions: Yes=True=Correct, No=False=Incorrect
#    - But if the standard answer is specific data, "Yes" is NOT equivalent to that data

# 4. Extract the answer:
#    - If "Final Answer:" is present, use content after it
#    - Ignore thinking process in <think> tags for final comparison
#    - Ignore case, minor formatting, extra whitespace

# 5. If the candidate's answer is invalid (incomplete, irrelevant, or cannot answer), select C (INVALID)

# Examples:
# - Standard: "800m" | Candidate: "Yes" → INCORRECT (wrong type)
# - Standard: "800m" | Candidate: "800m" → CORRECT
# - Standard: "2009, 5578 pts" | Candidate: "2009, 5578 pts" → CORRECT
# - Standard: "Yes" | Candidate: "Yes" → CORRECT
# - Standard: "No" | Candidate: "True" → INCORRECT

# Grade as:
# A: CORRECT - Answer matches the standard answer
# B: INCORRECT - Answer doesn't match or is wrong type
# C: INVALID - No valid answer provided

# Return only "A", "B", or "C".

# <Table Context>:
# {table}
# <Question>:
# {question}
# <Standard Answer>:
# {answer}
# <Candidate's Answer>:
# {prediction}

# Judging:
# """.strip()

# # ===== Fact Checking =====
# # ===== Fact Checking =====
# tablebench_fact_infer_cfg = dict(
#     prompt_template=dict(
#         type=PromptTemplate,
#         template=dict(
#             round=[
#                 dict(
#                     role='HUMAN',
#                     prompt="""{instruction}

# Table:
# {table}

# Question: {question}

# Please analyze the table carefully and provide your answer based on the information in the table. 

# Instructions:
# - If the question asks for specific information (numbers, names, dates, etc.), provide that specific information
# - If the question asks for verification (is/are, does/do, etc.), answer with Yes/No or True/False
# - Be precise and concise
# - End your response with "Final Answer: <your answer>"

# Answer:"""
#                 ),
#             ],
#         ),
#     ),
#     retriever=dict(type=ZeroRetriever),
#     inferencer=dict(type=GenInferencer, max_out_len=16384),  # 增加 max_out_len
# )

# tablebench_fact_eval_cfg = dict(
#     evaluator=dict(
#         type=GenericLLMEvaluator,
#         prompt_template=dict(
#             type=PromptTemplate,
#             template=dict(
#                 begin=[
#                     dict(
#                         role='SYSTEM',
#                         fallback_role='HUMAN',
#                         prompt="You are a helpful assistant who evaluates the correctness of fact-checking outputs for table understanding tasks.",
#                     )
#                 ],
#                 round=[
#                     dict(role='HUMAN', prompt=GRADER_TEMPLATE_FACT_CHECKING),
#                 ],
#             ),
#         ),
#         dataset_cfg=dict(
#             type=TableBenchDataset,
#             path=TABLEBENCH_HF_PATH,
#             qtype='FactChecking',  # ✅ 添加过滤条件
#             instruction_type='TCoT',  # ✅ 添加过滤条件
#             reader_cfg=tablebench_base_reader_cfg,
#         ),
#         judge_cfg=dict(),
#         dict_postprocessor=dict(type=generic_llmjudge_postprocess),
#     ),
# )

# # ===== Dataset Definitions =====
# tablebench_fact_checking_datasets = [
#     dict(
#         abbr='tablebench_fact_checking',
#         type=TableBenchDataset,
#         path=TABLEBENCH_HF_PATH,
#         qtype='FactChecking',
#         instruction_type='TCoT',
#         reader_cfg=tablebench_base_reader_cfg,
#         infer_cfg=tablebench_fact_infer_cfg,
#         eval_cfg=tablebench_fact_eval_cfg,
#     )
# ]

# # ===== Fact Checking =====
# # tablebench_fact_reader_cfg = tablebench_base_reader_cfg.copy()

# # tablebench_fact_infer_cfg = dict(
# #     prompt_template=dict(
# #         type=PromptTemplate,
# #         template=dict(
# #             round=[
# #                 dict(
# #                     role='HUMAN',
# #                     prompt="""Given the table below, verify the statement.

# # Table:
# # {table}

# # Statement: {question}

# # Please verify if this statement is correct based on the table.

# # Answer:"""
# #                 ),
# #             ],
# #         ),
# #     ),
# #     retriever=dict(type=ZeroRetriever),
# #     inferencer=dict(type=GenInferencer, max_out_len=256),
# # )

# # tablebench_fact_eval_cfg = dict(
# #     evaluator=dict(type=TableBenchEvaluator, metric='exact_match')
# # )

# # # ===== Dataset Definitions =====
# # tablebench_fact_checking_datasets = []

# # # FactChecking 类型的任务
# # tablebench_fact_checking_datasets.append(
# #     dict(
# #         abbr='tablebench_fact_checking',
# #         type=TableBenchDataset,
# #         path=TABLEBENCH_HF_PATH,
# #         qtype='FactChecking',
# #         reader_cfg=tablebench_fact_reader_cfg,
# #         infer_cfg=tablebench_fact_infer_cfg,
# #         eval_cfg=tablebench_fact_eval_cfg,
# #     )
# # )

"""TableBench Fact Checking任务配置
qtype='FactChecking' - 使用 Exact Match (EM) 评估
基于官方评测标准
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

Please analyze the table carefully and provide your answer based on the information in the table. 

Instructions:
- If the question asks for specific information (numbers, names, dates, etc.), provide that specific information
- If the question asks for verification (is/are, does/do, etc.), answer with Yes/No or True/False
- Be precise and concise
- End your response with "Final Answer: <your answer>"

Answer:"""
                ),
            ],
        ),
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer, max_out_len=4096),
)

# ===== 使用 Exact Match 评估器（符合官方标准）=====
tablebench_fact_eval_cfg = dict(
    evaluator=dict(
        type=TableBenchEvaluator,
        metric='exact_match_with_final_answer'  # 使用 EM，并提取 Final Answer
    )
)

# ===== Dataset Definitions =====
tablebench_fact_checking_datasets = [
    dict(
        abbr='tablebench_fact_checking',
        type=TableBenchDataset,
        path=TABLEBENCH_HF_PATH,
        qtype='FactChecking',
        instruction_type='TCoT',
        reader_cfg=tablebench_base_reader_cfg,
        infer_cfg=tablebench_fact_infer_cfg,
        eval_cfg=tablebench_fact_eval_cfg,
    )
]