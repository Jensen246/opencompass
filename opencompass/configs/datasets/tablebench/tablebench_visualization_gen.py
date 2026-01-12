"""TableBench Visualization任务配置
qtype='Visualization' 的各种子任务
"""
from mmengine.config import read_base

with read_base():
    from .tablebench_gen_base import (
        TABLEBENCH_HF_PATH,
        tablebench_base_reader_cfg,
        TableBenchDataset,
        TableBenchVisualizationEvaluator,
        PromptTemplate,
        ZeroRetriever,
        GenInferencer,
    )

from opencompass.datasets import generic_llmjudge_postprocess
from opencompass.evaluator import GenericLLMEvaluator


# LLM-based judge
# GRADER_TEMPLATE_VISUALIZATION = """
# Please as a grading expert, judge whether the candidate's Python visualization code correctly extracts the data arrays.

# Evaluation criteria:
# 1. **Standard Answer Format**:
#    - y_references = [[data_series_1], [data_series_2], ...]
#    - Each inner list represents one data series (e.g., one line in line chart, one group of bars)
#    - Example: y_references = [[24, 30, 36, 36, 35, 40, 44, 43, 41, 36, 32, 26]]
#    - This is a list with 1 data series containing 12 monthly values

# 2. **Core Evaluation Task**:
#    - Check if the candidate's code would extract the SAME data arrays as y_references
#    - Focus on DATA EXTRACTION logic, not code style or chart aesthetics
#    - The code should select the correct rows and columns from the table
#    - The code should extract the correct numerical values

# 3. Data Array Comparions:
#     - Standard answer format: y_references = [[data_series_1], [data_series_2], ...]
#     - Each inner list is one data series (one line, one set of bars, etc.)
#     - Example: y_references = [[24, 30, 36, 36, 35, 40, 44, 43, 41, 36, 32, 26]]
#     - Check if candidate's code would generate the same data arrays as the standard answer.
#     - Code must extract the correct values to match y_references format. 
    
# 4. Accept Valid Implementations:
#     - Different code styles are acceptable (plt.plot vs ax.plot)
#     - Different data extraction methods are acceptable
#     - Key requirement: The data arrays must match y_references format.
# 5. Common Requirements:
#     - For monthly data: should extract all 12 months (Jan-Dec)
#     - For unit conversion: should use correct units (inches, mm, etc.)
#     - For data filtering: should select correct rows (top N, specific categories)
# 6. Invalid Cases:
#     - No code provided
#     - Major syntax errors
#     - Wrong chart type
#     - Wrong data selection (wrong rows, columns, or values)

# Grade as:
# CORRECT: if the code is valid and generates the correct data arrays
# INCORRECT: if the code is invalid or generates incorrect data arrays
# INVALID: if no code is provided or there are major syntax errors

# Return only "A", "B", or "C".
# <Table>:
# {table}
# <Question>:
# {question}
# <Standard Answer (y_references)>:
# {answer}
# <Candidate's Code>:
# {prediction}
# Judging:
# """.strip()




# ===== Visualization =====
tablebench_viz_reader_cfg = tablebench_base_reader_cfg.copy()

tablebench_viz_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(
            round=[
                dict(
                    role='HUMAN',
                    prompt="""Generate Python code to create the requested visualization from the table below.

Table:
{table}

Question: {question}

Instructions: {instruction}

Important:
- The table will be saved as 'table.csv' with the first column as row labels
- Use pd.read_csv('table.csv') to load the data
- The first column contains row names (like "Record high °F (°C)")
- Use df.iloc[row_index, 1:] or df[df.iloc[:, 0] == 'row_name'] to access rows

Required code structure:
```python
import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv('table.csv')
...
plt.show()
```
Provide only the Python code without any explanation.
Answer:"""
                ),
            ],
        ),
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer, max_out_len=16384),
)

tablebench_viz_eval_cfg = dict(
    evaluator = dict(
        type = TableBenchVisualizationEvaluator,
        timeout = 10,
    )
)

# tablebench_viz_eval_cfg = dict(
#     evaluator=dict(
#         type=TableBenchVisualizationEvaluator,
#         prompt_template=dict(
#             type=PromptTemplate,
#             template=dict(
#                 begin=[
#                     dict(
#                         role='SYSTEM',
#                         fallback_role='HUMAN',
#                         prompt="You are a helpful assistant who evaluates the correctness and quality of models' outputs for table understanding tasks.",
#                     )
#                 ],
#                 round=[
#                     dict(
#                         role='HUMAN',
#                         prompt=GRADER_TEMPLATE_VISUALIZATION,
#                     )
#                 ],
#             ),
#         ),
#         dataset_cfg=dict(
#             type=TableBenchDataset,
#             path=TABLEBENCH_HF_PATH,
#             qtype='Visualization',
#             instruction_type='SCoT',
#             reader_cfg=tablebench_base_reader_cfg
#         ),
#         judge_cfg=dict(),
#         dict_postprocessor=dict(type=generic_llmjudge_postprocess),
#     ),
# )

# ===== Dataset Definitions =====
tablebench_visualization_datasets = []
tablebench_visualization_datasets.append(
    dict(
    abbr='tablebench_visualization',
    type=TableBenchDataset,
    path=TABLEBENCH_HF_PATH,
    qtype='Visualization',
    instruction_type='TCoT',
    reader_cfg=tablebench_viz_reader_cfg,
    infer_cfg=tablebench_viz_infer_cfg,
    eval_cfg=tablebench_viz_eval_cfg,
    )
)