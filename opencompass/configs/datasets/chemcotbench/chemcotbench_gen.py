"""ChemCoTBench Aggregated Dataset Configuration.

This configuration file aggregates all ChemCoTBench tasks:
- mol_und: Molecule Understanding (functional groups, scaffolds, SMILES equivalence)
- mol_edit: Molecule Editing (add, delete, substitute functional groups)
- mol_opt: Molecule Optimization (target-level and physicochemical-level)
- reaction: Chemical Reactions (forward, retro, conditions, NEPP, mechanism selection)

Note: This dataset requires HuggingFace access token.
Set environment variable HF_TOKEN or HUGGING_FACE_HUB_TOKEN before running.

Usage:
    # Set HuggingFace token
    export HF_TOKEN=your_huggingface_token

    # Run all ChemCoTBench tasks
    python run.py --datasets chemcotbench_gen --models your_model

    # Run specific task category
    python run.py --datasets chemcotbench_mol_und_gen --models your_model
    python run.py --datasets chemcotbench_mol_edit_gen --models your_model
    python run.py --datasets chemcotbench_mol_opt_gen --models your_model
    python run.py --datasets chemcotbench_reaction_gen --models your_model

Reference: https://huggingface.co/datasets/OpenMol/ChemCoTBench
Paper: https://arxiv.org/abs/2505.21318
"""

from mmengine.config import read_base

with read_base():
    from .chemcotbench_mol_und_gen import chemcotbench_mol_und_datasets
    from .chemcotbench_mol_edit_gen import chemcotbench_mol_edit_datasets
    from .chemcotbench_mol_opt_gen import chemcotbench_mol_opt_datasets
    from .chemcotbench_reaction_gen import chemcotbench_reaction_datasets

# Aggregate all datasets
chemcotbench_datasets = (
    chemcotbench_mol_und_datasets +
    chemcotbench_mol_edit_datasets +
    chemcotbench_mol_opt_datasets +
    chemcotbench_reaction_datasets
)
