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

# ===== Visualization =====
tablebench_viz_reader_cfg = tablebench_base_reader_cfg.copy()

tablebench_viz_infer_cfg = dict(
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

Please provide Python code using matplotlib to create the visualization. Wrap your code in ... ``` blocks.

Answer:"""
                ),
            ],
        ),
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer, max_out_len=1024),
)

tablebench_viz_eval_cfg = dict(
    evaluator=dict(type=TableBenchVisualizationEvaluator, timeout=15)
)

# ===== Dataset Definitions =====
tablebench_visualization_datasets = []

tablebench_visualization_datasets.append(
    dict(
        abbr='tablebench_visualization',
        type=TableBenchDataset,
        path=TABLEBENCH_HF_PATH,
        qtype='Visualization',
        instruction_type=None,  # 明确指定
        reader_cfg=tablebench_viz_reader_cfg,
        infer_cfg=tablebench_viz_infer_cfg,
        eval_cfg=tablebench_viz_eval_cfg,
    )
)## 主要修改
