"""PANORAMA PI4PC (Paragraph Identification) Dataset Configuration - CoT Mode.

This configuration file sets up the PI4PC task with Chain-of-Thought prompting
for evaluation in OpenCompass.

Usage:
    python run.py --datasets panorama_pi4pc_cot_gen --models your_model
"""

from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever

from opencompass.datasets.panorama.panorama_pi4pc import (
    PI4PCDataset,
    PI4PCEvaluator,
)

# Reader configuration
pi4pc_cot_reader_cfg = dict(
    input_columns=['prompt'],
    output_column='gold_answers',
)

# Inference configuration - longer output for CoT reasoning
pi4pc_cot_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template='{prompt}',
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(
        type=GenInferencer,
        max_out_len=16384,  # Longer output for CoT reasoning (includes mapping tables)
    ),
)

# Evaluation configuration
pi4pc_cot_eval_cfg = dict(
    evaluator=dict(type=PI4PCEvaluator),
)

# Dataset configuration - CoT mode
panorama_pi4pc_cot_datasets = [
    dict(
        abbr='panorama_pi4pc_cot',
        type=PI4PCDataset,
        path='LG-AI-Research/PANORAMA',
        prompt_mode='cot',
        reader_cfg=pi4pc_cot_reader_cfg,
        infer_cfg=pi4pc_cot_infer_cfg,
        eval_cfg=pi4pc_cot_eval_cfg,
    )
]
