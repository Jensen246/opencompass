# """TableBench Data Analysis任务配置
# qtype='DataAnalysis' 的各种子任务
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


# # ===== Statistical Analysis (Numeric) ====-
# GRADER_TEMPLATE_NUMERIC = """
# Please as a grading expert, judge whether the numerical answer given by the candidate is correct compared to the standard answer.

# Evaluation criteria:

# 1. **Answer Extraction - IMPORTANT**:
#    - Extract from "Final Answer:" marker if present
#    - **Only check the FINAL answer after "Final Answer:", NOT the thinking process**
#    - <think>...</think> tags are reasoning process, NOT the final answer

# 2. **Answer Type - CRITICAL**:
#    - Standard answer is numerical: Candidate MUST provide a number in Final Answer
#    - If candidate says "Yes/No/True/False" when a number is expected: INCORRECT
#    - Answer type must match (number vs. text vs. yes/no)

# 3. **Numerical Comparison**:
#    - Tolerance: ±0.01 for decimals, ±1% for percentages
#    - Format equivalence: "1000" = "1,000" = "1000.0"
#    - Unit equivalence: "0.5" = "50%" (when appropriate)

# 4. **Invalid Answers**:
#    - No Final Answer provided
#    - Incomplete or irrelevant response
#    - States cannot answer

# Examples:
# - Standard: "5578" | Candidate: "Final Answer: 5578" → A (CORRECT)
# - Standard: "5578" | Candidate: "<think>Found 5578</think> Final Answer: Yes" → B (INCORRECT - wrong type)
# - Standard: "800m" | Candidate: "Final Answer: Yes" → B (INCORRECT - should be 800m, not Yes)
# - Standard: "0.45" | Candidate: "Final Answer: 0.450" → A (CORRECT)
# - Standard: "2009, 5578 pts" | Candidate: "Final Answer: 2009, 5578 pts" → A (CORRECT)

# Grade as:
# A: CORRECT
# B: INCORRECT  
# C: INVALID

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

# GRADER_TEMPLATE_FACT_CHECKING = """
# Please as a grading expert, judge whether the candidate's answer is correct compared to the standard answer.

# Here are some evaluation criteria:
# 1. The standard answer is always correct. You only need to judge whether the candidate's answer matches the standard answer.
# 2. Extract the actual answer from the candidate's response:
#    - If there's a "Final Answer:" marker, use the content after it
#    - If there are <think>...</think> tags, check BOTH the thinking content AND the final output
#    - The answer might be in the reasoning process even if the final output is just "Yes" or "No"
# 3. For information extraction tasks:
#    - If the standard answer contains specific data (numbers, names, dates), check if the candidate correctly identified this information
#    - If the candidate says "Yes" but the standard answer is specific data, check if the reasoning contains the correct data
# 4. For binary questions (Yes/No, True/False):
#    - Accept equivalent expressions like Yes=True=Correct, No=False=Incorrect
# 5. Ignore case differences, minor formatting, and extra whitespace
# 6. If the candidate's answer is completely invalid (e.g., states it cannot answer, irrelevant content), select option C (INVALID)

# Please judge whether the following answers are consistent with the standard answer based on the above criteria. Grade the predicted answer as one of:
# A: CORRECT - The candidate correctly identified the information or gave the right answer
# B: INCORRECT - The candidate gave wrong information or missed the key details
# C: INVALID - The answer is unusable or irrelevant

# Just return the letters "A", "B", or "C", with no text around it.

# <Table Context>:
# {table}
# <Question>:
# {question}
# <Standard Answer>:
# {answer}
# <Candidate's Answer>:
# {prediction}

# Judging the correctness of the candidate's answer:
# """.strip()

# # ===== Statistical Analysis =====
# tablebench_statistical_reader_cfg = tablebench_base_reader_cfg.copy()

# tablebench_statistical_infer_cfg = dict(
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

# Please analyze the table and provide your answer. End your response with "Final Answer: <your detailed answer>".

# Answer:"""
#                 ),
#             ],
#         ),
#     ),
#     retriever=dict(type=ZeroRetriever),
#     inferencer=dict(type=GenInferencer, max_out_len=16384),
# )


# # ===== Data Analysis (general) =====
# # tablebench_statistical_eval_cfg = dict(
# #     evaluator=dict(type=TableBenchNumericEvaluator, tolerance=1e-2)
# # )

# tablebench_data_analysis_general_reader_cfg = tablebench_base_reader_cfg.copy()
# tablebench_data_analysis_general_infer_cfg = dict(
#     prompt_template = dict(
#         type = PromptTemplate,
#         template = dict(
#             round = [
#                 dict(
#                     role = 'HUMAN',
#                     prompt = """{instruction}

# Table:
# {table}

# Question: {question}

# Please analyze the table and provide your answer. End your response with "Final Answer: <your detailed answer>".

# Answer:"""
#                 ),
#             ],
#         ),
#     ),
#     retriever = dict(type = ZeroRetriever),
#     inferencer = dict(type = GenInferencer, max_out_len = 16384),
# )


# def create_llm_eval_cfg(grader_template: str):
#     """create LLM judge evaluation configuration"""
#     return dict(
#         evaluator = dict(
#             type = GenericLLMEvaluator,
#             prompt_template = dict(
#                 type = PromptTemplate,
#                 template = dict(
#                     begin = [
#                         dict(
#                             role = 'SYSTEM',
#                             fallback_role = 'HUMAN',
#                             prompt="You are a helpful assistant who evaluates the correctness and quality of models' outputs for table understanding tasks.",
#                         )
#                     ],
#                     round = [
#                         dict(
#                             role = 'HUMAN',
#                             prompt = grader_template,
#                         )
#                     ],
#                 ),
#             ),
#             dataset_cfg = dict(
#                 type = TableBenchDataset,
#                 path = TABLEBENCH_HF_PATH,
#                 qtype = 'DataAnalysis',
#                 instruction_type='TCoT',
#                 reader_cfg = tablebench_base_reader_cfg,
#                 ),
#             judge_cfg = dict(),
#             dict_postprocessor = dict(type = generic_llmjudge_postprocess),
#         ),
#     )

# # dataset definition
# tablebench_data_analysis_datasets = []

# #Numeric subtypes using numeric grader
# numeric_subtypes = ['CorrelationAnalysis','TrendForecasting','ImpactAnalysis']
# for subtype in numeric_subtypes:
#     eval_cfg = create_llm_eval_cfg(GRADER_TEMPLATE_NUMERIC)
#     eval_cfg['evaluator']['dataset_cfg']['qsubtype'] = subtype
#     tablebench_data_analysis_datasets.append(
#         dict(
#             abbr = f'tablebench_analysis_{subtype.lower()}',
#             type = TableBenchDataset,
#             path = TABLEBENCH_HF_PATH,
#             qtype = 'DataAnalysis',
#             qsubtype = subtype,
#             instruction_type = 'TCoT',
#             reader_cfg = tablebench_base_reader_cfg,
#             infer_cfg = tablebench_statistical_infer_cfg,
#             eval_cfg = eval_cfg,
#         )
#     )

# # eval_cfg = create_llm_eval_cfg(GRADER_TEMPLATE_DESCRIPTIVE)
# # eval_cfg['evaluator']['dataset_cfg']['qsubtype'] = 'ImpactAnalysis'

# # tablebench_data_analysis_datasets.append(
# #     dict(
# #         abbr='tablebench_analysis_ImpactAnalysis',
# #         type=TableBenchDataset,
# #         path=TABLEBENCH_HF_PATH,
# #         qtype='DataAnalysis',
# #         qsubtype='ImpactAnalysis',
# #         instruction_type='DP',
# #         reader_cfg=tablebench_base_reader_cfg,
# #         infer_cfg=tablebench_data_analysis_general_infer_cfg,
# #         eval_cfg=eval_cfg,
# #     )
# # )




# # tablebench_data_analysis_general_eval_cfg = dict(
# #     evaluator=dict(type=TableBenchEvaluator, metric='exact_match')
# # )

# # ===== Dataset Definitions =====
# # tablebench_data_analysis_datasets = []

# # # 定义 DataAnalysis 类型的各种子任务
# # # 格式: (qsubtype, abbr_suffix, reader_cfg, infer_cfg, eval_cfg)
# # data_analysis_tasks = [
# #     ('StatisticalAnalysis', 'stat', tablebench_statistical_reader_cfg, tablebench_statistical_infer_cfg, tablebench_statistical_eval_cfg),
# #     # 如果没有 qsubtype 过滤，加载所有 DataAnalysis 任务
# #     (None, 'all', tablebench_data_analysis_general_reader_cfg, tablebench_data_analysis_general_infer_cfg, tablebench_data_analysis_general_eval_cfg),
# # ]

# # for qsubtype, abbr_suffix, reader_cfg, infer_cfg, eval_cfg in data_analysis_tasks:
# #     dataset_cfg = dict(
# #         abbr=f'tablebench_analysis_{abbr_suffix}',
# #         type=TableBenchDataset,
# #         path=TABLEBENCH_HF_PATH,
# #         qtype='DataAnalysis',
# #         reader_cfg=reader_cfg,
# #         infer_cfg=infer_cfg,
# #         eval_cfg=eval_cfg,
# #     )
    
# #     # 只有当 qsubtype 不是 None 时才添加
# #     if qsubtype:
# #         dataset_cfg['qsubtype'] = qsubtype
    
# #     tablebench_data_analysis_datasets.append(dataset_cfg)



#official method
"""TableBench Data Analysis任务配置
qtype='DataAnalysis' 的各种子任务
基于官方评测标准
"""
from mmengine.config import read_base

with read_base():
    from .tablebench_gen_base import (
        TABLEBENCH_HF_PATH,
        tablebench_base_reader_cfg,
        TableBenchDataset,
        TableBenchEvaluator,
        TableBenchNumericalWithPercenteErrorEvaluator,  # 新增
        TableBenchRougeEvaluator,  # 新增
        PromptTemplate,
        ZeroRetriever,
        GenInferencer,
    )

# ===== Inference Configuration =====
tablebench_data_analysis_infer_cfg = dict(
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

Please analyze the table and provide your answer. End your response with "Final Answer: <your detailed answer>".

Answer:"""
                ),
            ],
        ),
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer, max_out_len=4096),
)

# ===== Dataset Definitions =====
tablebench_data_analysis_datasets = []

# 1. CorrelationAnalysis - 使用 EM_with_error_10 (10% 误差容忍)
tablebench_data_analysis_datasets.append(
    dict(
        abbr='tablebench_analysis_correlationanalysis',
        type=TableBenchDataset,
        path=TABLEBENCH_HF_PATH,
        qtype='DataAnalysis',
        qsubtype='CorrelationAnalysis',
        instruction_type='TCoT',  # 或根据需要选择 DP/TCoT/SCoT/PoT
        reader_cfg=tablebench_base_reader_cfg,
        infer_cfg=tablebench_data_analysis_infer_cfg,
        eval_cfg=dict(
            evaluator=dict(
                type=TableBenchNumericalWithPercenteErrorEvaluator,
                error_rate=0.10  # 10% tolerance
            )
        ),
    )
)

# 2. TrendForecasting - 使用 EM_with_error_10
tablebench_data_analysis_datasets.append(
    dict(
        abbr='tablebench_analysis_trendforecasting',
        type=TableBenchDataset,
        path=TABLEBENCH_HF_PATH,
        qtype='DataAnalysis',
        qsubtype='TrendForecasting',
        instruction_type='TCoT',
        reader_cfg=tablebench_base_reader_cfg,
        infer_cfg=tablebench_data_analysis_infer_cfg,
        eval_cfg=dict(
            evaluator=dict(
                type=TableBenchNumericalWithPercenteErrorEvaluator,
                error_rate=0.10
            )
        ),
    )
)

# 3. StatisticalAnalysis - 使用 EM_with_error_10
tablebench_data_analysis_datasets.append(
    dict(
        abbr='tablebench_analysis_statisticalanalysis',
        type=TableBenchDataset,
        path=TABLEBENCH_HF_PATH,
        qtype='DataAnalysis',
        qsubtype='StatisticalAnalysis',
        instruction_type='TCoT',
        reader_cfg=tablebench_base_reader_cfg,
        infer_cfg=tablebench_data_analysis_infer_cfg,
        eval_cfg=dict(
            evaluator=dict(
                type=TableBenchNumericalWithPercenteErrorEvaluator,
                error_rate=0.10
            )
        ),
    )
)

# 4. ImpactAnalysis - 使用 EM (精确匹配)
tablebench_data_analysis_datasets.append(
    dict(
        abbr='tablebench_analysis_impactanalysis',
        type=TableBenchDataset,
        path=TABLEBENCH_HF_PATH,
        qtype='DataAnalysis',
        qsubtype='ImpactAnalysis',
        instruction_type='TCoT',
        reader_cfg=tablebench_base_reader_cfg,
        infer_cfg=tablebench_data_analysis_infer_cfg,
        eval_cfg=dict(
            evaluator=dict(
                type=TableBenchEvaluator,
                metric='exact_match_with_final_answer'  # 精确匹配
            )
        ),
    )
)

# 5. 其他子类型 - 使用 ROUGE-L
# 如果有其他子类型，可以添加如下配置：
tablebench_data_analysis_datasets.append(
    dict(
        abbr='tablebench_analysis_other',
        type=TableBenchDataset,
        path=TABLEBENCH_HF_PATH,
        qtype='DataAnalysis',
        instruction_type='TCoT',
        reader_cfg=tablebench_base_reader_cfg,
        infer_cfg=tablebench_data_analysis_infer_cfg,
        eval_cfg=dict(
            evaluator=dict(
                type=TableBenchRougeEvaluator,
            )
        ),
    )
)