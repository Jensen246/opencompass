"""PANORAMA NOC4PC (Novelty/Obviousness Classification) Dataset Configuration - CoT Mode.

This configuration file sets up the NOC4PC task with Chain-of-Thought prompting
for evaluation in OpenCompass.

Usage:
    python run.py --datasets panorama_noc4pc_cot_gen --models your_model
"""

from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever

from opencompass.datasets.panorama.panorama_noc4pc import (
    NOC4PCDataset,
    NOC4PCEvaluator,
)

# Reader configuration
noc4pc_cot_reader_cfg = dict(
    input_columns=['prompt'],
    output_column='gold_code',
)

# Inference configuration - longer output for CoT reasoning
noc4pc_cot_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template='{prompt}',
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(
        type=GenInferencer,
        max_out_len=16384,  # Longer output for CoT reasoning
    ),
)

# Evaluation configuration
noc4pc_cot_eval_cfg = dict(
    evaluator=dict(type=NOC4PCEvaluator),
)

# Dataset configuration - CoT mode
panorama_noc4pc_cot_datasets = [
    dict(
        abbr='panorama_noc4pc_cot',
        type=NOC4PCDataset,
        path='LG-AI-Research/PANORAMA',
        prompt_mode='cot',
        max_input_len=16384,
        tokenizer_path='Qwen/Qwen2.5-7B-Instruct',
        reader_cfg=noc4pc_cot_reader_cfg,
        infer_cfg=noc4pc_cot_infer_cfg,
        eval_cfg=noc4pc_cot_eval_cfg,
    )
]
