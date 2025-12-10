"""PANORAMA PAR4PC (Prior Art Retrieval) Dataset Configuration.

This configuration file sets up the PAR4PC task for evaluation in OpenCompass.
The task involves identifying which patents (A-H) were cited as prior art
for a given claim rejection.

Usage:
    python run.py --datasets panorama_par4pc_gen --models your_model
"""

from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever

from opencompass.datasets.panorama.panorama_par4pc import (
    PAR4PCDataset,
    PAR4PCEvaluator,
)

# Reader configuration
par4pc_reader_cfg = dict(
    input_columns=['prompt'],
    output_column='gold_answers',
)

# Inference configuration
par4pc_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template='{prompt}',
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(
        type=GenInferencer,
        max_out_len=8192,
    ),
)

# Evaluation configuration
par4pc_eval_cfg = dict(
    evaluator=dict(type=PAR4PCEvaluator),
)

# Dataset configuration - zero-shot mode
panorama_par4pc_datasets = [
    dict(
        abbr='panorama_par4pc',
        type=PAR4PCDataset,
        path='LG-AI-Research/PANORAMA',
        prompt_mode='zero-shot',
        reader_cfg=par4pc_reader_cfg,
        infer_cfg=par4pc_infer_cfg,
        eval_cfg=par4pc_eval_cfg,
    )
]
