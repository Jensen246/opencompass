"""ChemCoTBench Molecule Understanding Zero-shot Configuration.

Evaluates:
- fg_count: Functional group counting
- ring_count: Ring counting
- Murcko_scaffold: Murcko scaffold identification
- ring_system_scaffold: Ring system scaffold identification
- equivalence: SMILES equivalence checking
"""

from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer

# Dataset configuration
_mol_und_subtasks = ['fg_count', 'ring_count', 'Murcko_scaffold', 'ring_system_scaffold', 'equivalence']

chemcotbench_mol_und_datasets = []

for subtask in _mol_und_subtasks:
    reader_cfg = dict(
        input_columns=['prompt'],
        output_column='answer',
    )

    infer_cfg = dict(
        prompt_template=dict(
            type=PromptTemplate,
            template=dict(
                round=[
                    dict(role='HUMAN', prompt='{prompt}'),
                ]
            ),
        ),
        retriever=dict(type=ZeroRetriever),
        inferencer=dict(type=GenInferencer, max_out_len=1024),
    )

    eval_cfg = dict(
        evaluator=dict(
            type='ChemCoTBenchMolUndEvaluator',
            subtask=subtask,
        ),
    )

    chemcotbench_mol_und_datasets.append(
        dict(
            abbr=f'chemcotbench_mol_und_{subtask}',
            type='ChemCoTBenchMolUndDataset',
            path='OpenMol/ChemCoTBench',
            subtask=subtask,
            reader_cfg=reader_cfg,
            infer_cfg=infer_cfg,
            eval_cfg=eval_cfg,
        )
    )
