from opencompass.datasets import NvBench2Dataset, NvBench2Evaluator
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever

# 官方 Basic Prompt（来自 basic_prompt.py 第117行）
# combined_input = sys_content + simple_prompt + example_vl + input
# sys_content = "You are an intelligent assistant. You only answer with #OUTPUT."
BASIC_PROMPT = '''You are an intelligent assistant. You only answer with #OUTPUT.
You are a good data visualization expert. Given an ambiguous/incomplete Natural Language Query and a Data Table, please recommend 1 to 5 different charts corresponding for the ambiguous/incomplete NL Query. Ensure to strictly follow the output format.
# Output format(JSON array):
#OUTPUT:[
{{vega chart 1}},
{{vega chart 2}},
 ...]

# EXAMPLE Vega-Lite Chart: (mark can be bar, line, arc, point, rect, boxplot)
## e.g.1 a bar chart with average sale number over binned price, bin num is 10, filter by date > year 2000.
- {{"mark": "bar", "encoding": {{"x": {{"field": "price", "bin": {{"maxbins":10}}}}, "y": {{"field": "sale_number", "aggregate": "mean"}}}}, "transform": [{{"filter": {{"field": "date", "gte": {{"year": 2000}}}}}}]}}
## e.g.2 a pie chart with average price over area, filter by product type is notebook or pencil.
- {{"mark": "arc", "encoding": {{"color": {{"field": "area"}}, "theta": {{"field": "price", "aggregate": "mean"}}}}, "transform": [{{"filter": {{"field": "product_type", "oneOf": ["notebook", "pencil"]}}}}]}}
## e.g.3 a heatmap with date on x (binned by year), area on y, sum of sale number on color, filter by 120 <= price <= 200.
- {{"mark": "rect", "encoding": {{"x": {{"field": "date", "timeUnit": "year"}}, "y": {{"field": "area"}}, "color": {{"field": "sale_number", "aggregate": "sum"}}}}, "transform": [{{"filter": {{"field": "price", "range": [120, 200]}}}}]}}
## e.g.4 a line chart showing the count of products over product categories, filter by date < year 2000, and sort by the count of products in descending order.
- {{"mark": "line", "encoding": {{"x": {{"field": "product", "sort": "-y"}}, "y": {{"aggregate": "count"}}}}, "transform": [{{"filter": {{"field": "date", "lte": {{"year": 2000}}}}}}]}}

#INPUT:
Data Schema: {data_schema}
Value Example: {value_example}
Unique Value Num: {unique_value_num}
NL Query: {nl_query}
'''

nvbench2_basic_reader_cfg = dict(
    input_columns=['nl_query', 'data_schema', 'value_example', 'unique_value_num'],
    output_column='gold_answer',
)

nvbench2_basic_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(
            round=[
                dict(role='HUMAN', prompt=BASIC_PROMPT),
            ],
        ),
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer, max_out_len=4096),
)

nvbench2_basic_eval_cfg = dict(
    evaluator=dict(type=NvBench2Evaluator, k_values=[1, 3, 5]),
    pred_role='BOT',
)

nvbench2_basic_datasets = [
    dict(
        type=NvBench2Dataset,
        abbr='nvbench2_basic',
        path='TianqiLuo/nvBench2.0',
        split='test',
        reader_cfg=nvbench2_basic_reader_cfg,
        infer_cfg=nvbench2_basic_infer_cfg,
        eval_cfg=nvbench2_basic_eval_cfg,
    )
]
