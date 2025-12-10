"""PANORAMA NOC4PC (Novelty/Obviousness Classification) Dataset Configuration.

This configuration file sets up the NOC4PC task for evaluation in OpenCompass.
The task involves classifying patent claims as ALLOW, 102 (novelty rejection),
or 103 (obviousness rejection).

Usage:
    python run.py --datasets panorama_noc4pc_gen --models your_model
"""

from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever

from opencompass.datasets.panorama.panorama_noc4pc import (
    NOC4PCDataset,
    NOC4PCEvaluator,
)

# Reader configuration
noc4pc_reader_cfg = dict(
    input_columns=['prompt'],
    output_column='gold_code',
)

# Inference configuration
noc4pc_infer_cfg = dict(
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
noc4pc_eval_cfg = dict(
    evaluator=dict(type=NOC4PCEvaluator),
)

# Dataset configuration - zero-shot mode
panorama_noc4pc_datasets = [
    dict(
        abbr='panorama_noc4pc',
        type=NOC4PCDataset,
        path='LG-AI-Research/PANORAMA',
        prompt_mode='zero-shot',
        reader_cfg=noc4pc_reader_cfg,
        infer_cfg=noc4pc_infer_cfg,
        eval_cfg=noc4pc_eval_cfg,
    )
]
