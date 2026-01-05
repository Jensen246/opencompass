"""ChemCoTBench Datasets and Evaluators.

ChemCoTBench is a benchmark for evaluating LLMs on chemical reasoning tasks:
- mol_und: Molecule Understanding (functional groups, scaffolds, SMILES equivalence)
- mol_edit: Molecule Editing (add, delete, substitute functional groups)
- mol_opt: Molecule Optimization (target-level and physicochemical-level)
- reaction: Chemical Reactions (forward, retro, conditions, NEPP, mechanism selection)

Reference: https://huggingface.co/datasets/OpenMol/ChemCoTBench
Paper: https://arxiv.org/abs/2505.21318
"""

from .chemcotbench_mol_und import (
    ChemCoTBenchMolUndDataset,
    ChemCoTBenchMolUndEvaluator,
)
from .chemcotbench_mol_edit import (
    ChemCoTBenchMolEditDataset,
    ChemCoTBenchMolEditEvaluator,
)
from .chemcotbench_mol_opt import (
    ChemCoTBenchMolOptDataset,
    ChemCoTBenchMolOptEvaluator,
)
from .chemcotbench_reaction import (
    ChemCoTBenchReactionDataset,
    ChemCoTBenchReactionEvaluator,
)

__all__ = [
    # Molecule Understanding
    'ChemCoTBenchMolUndDataset',
    'ChemCoTBenchMolUndEvaluator',
    # Molecule Editing
    'ChemCoTBenchMolEditDataset',
    'ChemCoTBenchMolEditEvaluator',
    # Molecule Optimization
    'ChemCoTBenchMolOptDataset',
    'ChemCoTBenchMolOptEvaluator',
    # Reaction
    'ChemCoTBenchReactionDataset',
    'ChemCoTBenchReactionEvaluator',
]


