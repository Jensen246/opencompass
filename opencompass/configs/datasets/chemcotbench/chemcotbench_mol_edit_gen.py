"""ChemCoTBench Molecule Editing Zero-shot Configuration.

Evaluates:
- add: Adding functional groups
- delete: Deleting functional groups
- sub: Substituting functional groups
"""

from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer

# Dataset configuration
_mol_edit_subtasks = ['add', 'delete', 'sub']

chemcotbench_mol_edit_datasets = []

for subtask in _mol_edit_subtasks:
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
            type='ChemCoTBenchMolEditEvaluator',
            subtask=subtask,
        ),
    )

    chemcotbench_mol_edit_datasets.append(
        dict(
            abbr=f'chemcotbench_mol_edit_{subtask}',
            type='ChemCoTBenchMolEditDataset',
            path='OpenMol/ChemCoTBench',
            subtask=subtask,
            reader_cfg=reader_cfg,
            infer_cfg=infer_cfg,
            eval_cfg=eval_cfg,
        )
    )
