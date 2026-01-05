"""TableBench Data Analysis任务配置
qtype='DataAnalysis' 的各种子任务
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

GRADER_TEMPLATE_NUMERIC = """
Please as a grading expert, judge whether the numerical answer given by the candidate is correct compared to the standard answer.

Here are some evaluation criteria:
1. The standard answer is always correct. You only need to judge whether the candidate's answer matches the standard answer.
2. For numerical answers, consider answers correct if they are within a reasonable tolerance (e.g., ±0.01 or ±1% for percentages).
3. Ignore formatting differences (e.g., "1000" vs "1,000", "0.5" vs "50%").
4. If the prediction contains "Final Answer:", extract the answer after this marker. Otherwise, try to extract the numerical value from the response.
5. If the candidate's answer is invalid (e.g., incomplete, irrelevant, or states it cannot answer), select option C (INVALID).

Please judge whether the following answers are consistent with the standard answer based on the above criteria. Grade the predicted answer as one of:
A: CORRECT 
B: INCORRECT
C: INVALID

Just return the letters "A", "B", or "C", with no text around it.

Here is your task. Simply reply with either CORRECT, INCORRECT, or INVALID. Don't apologize or correct yourself if there was a mistake; we are just trying to grade the answer.

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

GRADER_TEMPLATE_DESCRIPTIVE = """
Please as a grading expert, judge whether the descriptive answer given by the candidate is correct and comprehensive compared to the standard answer.

Here are some evaluation criteria:
1. The standard answer is always correct. Judge whether the candidate's answer conveys the same key information.
2. The candidate's answer does not need to match word-for-word, but should contain the main points and insights from the standard answer.
3. Consider answers correct if they capture the essential meaning, even with different wording or structure.
4. Ignore minor differences in phrasing, but check for factual accuracy based on the table data.
5. If the prediction contains "Final Answer:", extract the answer after this marker.
6. If the candidate's answer is invalid (e.g., incomplete, cut off mid-response, irrelevant, or states it cannot answer), select option C (INVALID).

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



# ===== Statistical Analysis =====
tablebench_statistical_reader_cfg = tablebench_base_reader_cfg.copy()

tablebench_statistical_infer_cfg = dict(
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

Please analyze the table and provide your answer. End your response with "Final Answer: <your answer>".

Answer:"""
                ),
            ],
        ),
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer, max_out_len=512),
)

tablebench_data_analysis_general_infer_cfg = dict(
    prompt_template = dict(
        type = PromptTemplate,
        template = dict(
            round = [
                dict(
                    role = 'HUMAN',
                    prompt = """{instruction}

Table:
{table}

Question: {question}

Please analyze the table and provide your answer. End your response with "Final Answer: <your detailed answer>".

Answer:"""
                ),
            ],
        ),
    ),
    retriever = dict(type = ZeroRetriever),
    inferencer = dict(type = GenInferencer, max_out_len = 512),
)

#===== Eval config with LLM judge =====
def create_llm_eval_cfg(grader_template):
    return dict(
        evaluator = dict
    )

# ===== Data Analysis (general) =====
tablebench_data_analysis_general_reader_cfg = tablebench_base_reader_cfg.copy()

tablebench_data_analysis_general_infer_cfg = dict(
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

Answer:"""
                ),
            ],
        ),
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer, max_out_len=512),
)

tablebench_data_analysis_general_eval_cfg = dict(
    evaluator=dict(type=TableBenchEvaluator, metric='f1')  # 改用 f1，更宽松
)

# ===== Dataset Definitions =====
# tablebench_data_analysis_datasets = []

# # 定义 DataAnalysis 类型的各种子任务
# data_analysis_tasks = [
#     ('StatisticalAnalysis', 'stat', tablebench_statistical_reader_cfg, tablebench_statistical_infer_cfg, tablebench_statistical_eval_cfg),
#     (None, 'all', tablebench_data_analysis_general_reader_cfg, tablebench_data_analysis_general_infer_cfg, tablebench_data_analysis_general_eval_cfg),
# ]

# for qsubtype, abbr_suffix, reader_cfg, infer_cfg, eval_cfg in data_analysis_tasks:
#     dataset_cfg = dict(
#         abbr=f'tablebench_analysis_{abbr_suffix}',
#         type=TableBenchDataset,
#         path=TABLEBENCH_HF_PATH,
#         qtype='DataAnalysis',
#         instruction_type='TCoT',  # 明确指定使用 DP (Direct Prompting)
#         reader_cfg=reader_cfg,
#         infer_cfg=infer_cfg,
#         eval_cfg=eval_cfg,
#     )
    
#     if qsubtype:
#         dataset_cfg['qsubtype'] = qsubtype
    
#     tablebench_data_analysis_datasets.append(dataset_cfg)


#==== Dataset Definitions ===
#1. need EM_with_error_10 to evaluate
tablebench_data_analysis_datasets = []
# 第 136-151 行
numeric_subtype = ['CorrelationAnalysis','TrendForecasting','StatisticalAnalysis']
for subtype in numeric_subtype:
    tablebench_data_analysis_datasets.append(
        dict(
            abbr = f'tablebench_data_analysis_{subtype.lower()}',
            type = TableBenchDataset,
            path = TABLEBENCH_HF_PATH,
            qtype = 'DataAnalysis',
            qsubtype = subtype, 
            instruction_type = 'DP',
            reader_cfg = tablebench_statistical_reader_cfg,
            infer_cfg = tablebench_statistical_infer_cfg,
            eval_cfg = dict(
                evaluator = dict(type = TableBenchNumericEvaluator, tolerance = 1e-3)
            ),
        )
    )


tablebench_data_analysis_datasets.append(
    dict(
        abbr = 'tablebench_analysis_impactanalysis',
        type = TableBenchDataset,
        path = TABLEBENCH_HF_PATH,
        qtype = 'DataAnalysis',
        qsubtype = 'ImpactAnalysis', 
        instruction_type = None,
        reader_cfg = tablebench_base_reader_cfg,
        infer_cfg = tablebench_data_analysis_general_infer_cfg,
        eval_cfg = dict(
            evaluator = dict(type = TableBenchRougeEvaluator)
        ),
    )
)




# tablebench_data_analysis_datasets.append(
#     dict(
#         abbr='tablebench_analysis_others',
#         type=TableBenchDataset,
#         path=TABLEBENCH_HF_PATH,
#         qtype='DataAnalysis',
#         # 不指定 qsubtype，会包含所有子任务（包括上面已经定义的）
#         # 如果要排除已定义的，需要在 TableBenchDataset 中添加过滤逻辑
#         instruction_type='TCoT',
#         reader_cfg=tablebench_data_analysis_general_reader_cfg,
#         infer_cfg=tablebench_data_analysis_general_infer_cfg,
#         eval_cfg=dict(
#             evaluator=dict(type=TableBenchEvaluator, metric='f1')  # 或者使用 ROUGE-L
#         ),
#     )
# )