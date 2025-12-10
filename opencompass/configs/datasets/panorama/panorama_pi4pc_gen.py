"""PANORAMA PI4PC (Paragraph Identification) Dataset Configuration.

This configuration file sets up the PI4PC task for evaluation in OpenCompass.
The task involves identifying the single most relevant paragraph from prior art
that supports the rejection of a patent claim.

Usage:
    python run.py --datasets panorama_pi4pc_gen --models your_model
"""

from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever

from opencompass.datasets.panorama.panorama_pi4pc import (
    PI4PCDataset,
    PI4PCEvaluator,
)

# Reader configuration
pi4pc_reader_cfg = dict(
    input_columns=['prompt'],
    output_column='gold_answers',
)

# Inference configuration
pi4pc_infer_cfg = dict(
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
pi4pc_eval_cfg = dict(
    evaluator=dict(type=PI4PCEvaluator),
)

# Dataset configuration - zero-shot mode
panorama_pi4pc_datasets = [
    dict(
        abbr='panorama_pi4pc',
        type=PI4PCDataset,
        path='LG-AI-Research/PANORAMA',
        prompt_mode='zero-shot',
        reader_cfg=pi4pc_reader_cfg,
        infer_cfg=pi4pc_infer_cfg,
        eval_cfg=pi4pc_eval_cfg,
    )
]
