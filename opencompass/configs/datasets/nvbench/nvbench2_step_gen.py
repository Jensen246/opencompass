from opencompass.datasets import NvBench2Dataset, NvBench2Evaluator
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever

# 官方 Step Prompt（来自 step_prompt.py 第267行）
# combined_input = sys_content + simple_prompt + simple_step_example + input
# sys_content = "You are an intelligent assistant."
STEP_PROMPT = '''You are an intelligent assistant.
You are a good data visualization expert. Given an ambiguous/incomplete Natural Language Query, a Data Table, your task is to to generate 1 to 5 different visualization charts corresponding for the ambiguous/incomplete NL Query. Please think step by step.
# Instructions:
Step 1: Extract Data Columns and Filters from NL Query.
- First, identify the relevant columns (fields) mentioned in the NL query.
- Second, identify the data filters (conditions) mentioned in the NL query if they exist. Otherwise, leave the filter answer as empty list.

Step 2: Extract Data Transformation from NL Query.
- Identify the data transformation mentioned in the NL Query.operations=(aggregat;bin;sort).

Step 3: Select Chart Type from NL Query.
- Possible chart type=(bar, line, arc, point, rect, boxplot). Note that 'arc' is identical to the pie chart, and 'rect' is identical to the heatmap chart.

Step 4: Chart Channel Mapping.
- This subtask maps the selected data columns (from step 1) and data transformations (from step 2) to the chosen chart types (from step 3), assigning data to appropriate visual encoding channels.
- Note that, there are obligatory or optional channels, follow the defination (chart:channel)=(bar:[x*,y*,color]; line:[x*,y*,color]; arc:[color*,theta*]; point:[x*,y*,color,size]; rect:[x*,y*,color*]; boxplot:[x*,y*]), where the channels with * obligatory, the channels without * are optional.
- Note: aggregation 'count' can be considered as a special computed data column to fill in Quantitative (Q) channel, and it must not include a `field`, e.g.{{"x": {{"field": "product"}}, "y": {{"aggregate": "count"}}}} means y is the counted row number for each product.
- you can consider chart with or without optional channels.
- Output a chart list of all possible channel mappings.

Step 5: Add implicit data channelmapping.
- Implicit data channel mapping means if the selected data columns are not enough to complete a chart's channels, you need to complete the chart channels with additional columns from data table.
- Output a chart list of all possible channel mappings.

Step 6: Add implicit data transformation and complete chart with data filters.
- Implicit data transformation refers to the transformations are not mentioned in the NL query, but are helpful to generate valid chart.
- First, add implicit transformation following basic feasibility rules (if needed):
-- For bar chart, if x is a quantitative column with too many numbers (>20), x should be binned.
-- For bar chart, if x have duplicated values or x is binned, then y should be aggregated.
-- (Other feasibility rules you think properly)
- Second, add data filter from step 1 (if exists) to complete the final chart list.
- Output a chart list of all possible channel mappings along with the filters.
--Here are some examples of Vega-Lite Chart: (mark can be bar, line, arc, point, rect, boxplot)
--- e.g.1 a bar chart with average sale number over binned price, bin num is 10, filter by date > year 2000: {{"mark": "bar", "encoding": {{"x": {{"field": "price", "bin": {{"maxbins":10}}}}, "y": {{"field": "sale_number", "aggregate": "mean"}}}}, "transform": [{{"filter": {{"field": "date", "gte": {{"year": 2000}}}}}}]}}

# Here is an example answer, but it is not complete, you need to fill in the reasoning process for each step.
Your output must strictly follow the JSON format shown in this example, with "thinking_steps" and "final_output" as the root level keys.
Note that final_output should generate 1 to 5 different visualization charts to cover the possible answer space for ambiguous/incomplete NL Query. Try to explore different combinations of channels and transformations to provide diverse and meaningful visualizations:
## INPUT:
Data Schema: ['player_id', 'birth_year', 'birth_month', 'birth_day', 'birth_country', 'birth_state', 'birth_city', 'death_year', 'death_month', 'death_day', 'death_country', 'death_state', 'death_city', 'weight', 'height']
Value Example: {{'player_id': ['aardsda01', 'muirjo01', 'zychto01'], 'birth_year': ['1970-01-01 00:00:00.000001820', '1970-01-01 00:00:00.000001955', '1970-01-01 00:00:00.000001995'], 'birth_month': ['1970-01-01 00:00:00.000000001', '1970-01-01 00:00:00.000000004', '1970-01-01 00:00:00.000000012'], 'birth_day': [1.0, 17.0, 31.0], 'birth_country': ['Afghanistan', 'Belize', 'Viet Nam'], 'birth_state': ['AB', 'Dodescanese Isl.', 'Zulia'], 'birth_city': ['Aberdeen', 'Childress', 'Zion'], 'death_year': ['1970-01-01 00:00:00.000001872', '1970-01-01 00:00:00.000001923', '1970-01-01 00:00:00.000002016'], 'death_month': ['1970-01-01 00:00:00.000000001', '1970-01-01 00:00:00.000000009', '1970-01-01 00:00:00.000000012'], 'death_day': [1.0, 29.0, 31.0], 'death_country': ['American Samoa', 'Cuba', 'Venezuela'], 'death_state': ['AB', 'TN', 'Zulia'], 'death_city': ['Aberdeen', 'Albemarle', 'Zimmerman'], 'weight': [65.0, 206.0, 320.0], 'height': [43.0, 67.0, 83.0]}}
Unique Value Num: {{'player_id': 18846, 'birth_year': 165, 'birth_month': 12, 'birth_day': 31, 'birth_country': 52, 'birth_state': 245, 'birth_city': 4713, 'death_year': 145, 'death_month': 12, 'death_day': 31, 'death_country': 23, 'death_state': 92, 'death_city': 2553, 'weight': 131, 'height': 22}}
NL Query: Please mark a scatter plot with columns for birth day, birth city, and day, aggregating the sum of the day while filtering for birth city within Neath, Farnhamville, and Tunstall, and filtering for death day less than or equal to 25.0.
## output:```json
{{
    "thining_steps":"You fill in the reasoning process for each step.",
    "final_output": [
        {{
            "mark": "point",
            "encoding": {{
                "x": {{"field": "birth_day"}},
                "color": {{"field": "birth_city"}},
                "size": {{"field": "death_day", "aggregate": "sum"}},
                "y": {{"field": "height"}}
            }},
            "transform": [
                {{"filter": {{"field": "birth_city", "oneOf": ["Neath", "Farnhamville", "Tunstall"]}}}},
                {{"filter": {{"field": "death_day", "lte": 25.0}}}}
            ]
        }},
        {{
            "mark": "point",
            "encoding": {{
                "x": {{"field": "birth_day"}},
                "color": {{"field": "birth_city"}},
                "size": {{"field": "death_day", "aggregate": "sum"}},
                "y": {{"field": "weight"}}
            }},
            "transform": [
                {{"filter": {{"field": "birth_city", "oneOf": ["Neath", "Farnhamville", "Tunstall"]}}}},
                {{"filter": {{"field": "death_day", "lte": 25.0}}}}
            ]
        }}
    ]
}}```

#INPUT:
Data Schema: {data_schema}
Value Example: {value_example}
Unique Value Num: {unique_value_num}
NL Query: {nl_query}
'''

nvbench2_step_reader_cfg = dict(
    input_columns=['nl_query', 'data_schema', 'value_example', 'unique_value_num'],
    output_column='gold_answer',
)

nvbench2_step_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(
            round=[
                dict(role='HUMAN', prompt=STEP_PROMPT),
            ],
        ),
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer, max_out_len=8192),
)

nvbench2_step_eval_cfg = dict(
    evaluator=dict(type=NvBench2Evaluator, k_values=[1, 3, 5]),
    pred_role='BOT',
)

nvbench2_step_datasets = [
    dict(
        type=NvBench2Dataset,
        abbr='nvbench2_step',
        path='TianqiLuo/nvBench2.0',
        split='test',
        reader_cfg=nvbench2_step_reader_cfg,
        infer_cfg=nvbench2_step_infer_cfg,
        eval_cfg=nvbench2_step_eval_cfg,
    )
]
