"""PANORAMA Benchmark Aggregated Dataset Configuration.

This configuration file aggregates all three PANORAMA tasks:
- PAR4PC: Prior Art Retrieval for Patent Claims
- PI4PC: Paragraph Identification for Patent Claims
- NOC4PC: Novelty/Obviousness Classification for Patent Claims

Each task is available in both zero-shot and CoT modes.

Usage:
    # Run all PANORAMA tasks (zero-shot)
    python run.py --datasets panorama_gen --models your_model

    # Run specific task
    python run.py --datasets panorama_par4pc_gen --models your_model
    python run.py --datasets panorama_pi4pc_gen --models your_model
    python run.py --datasets panorama_noc4pc_gen --models your_model

    # Run CoT versions
    python run.py --datasets panorama_par4pc_cot_gen --models your_model
    python run.py --datasets panorama_pi4pc_cot_gen --models your_model
    python run.py --datasets panorama_noc4pc_cot_gen --models your_model

Reference: https://huggingface.co/datasets/LG-AI-Research/PANORAMA
"""

from mmengine.config import read_base

with read_base():
    from .panorama_par4pc_gen import panorama_par4pc_datasets
    from .panorama_pi4pc_gen import panorama_pi4pc_datasets
    from .panorama_noc4pc_gen import panorama_noc4pc_datasets

# Aggregate all zero-shot datasets
panorama_datasets = (
    panorama_par4pc_datasets +
    panorama_pi4pc_datasets +
    panorama_noc4pc_datasets
)
