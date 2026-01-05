"""ChemCoTBench Molecule Optimization Zero-shot Configuration.

Evaluates:
- drd, gsk, jnk: Drug target optimization
- logp, qed, solubility: Physicochemical property optimization
"""

from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer

# Dataset configuration
_mol_opt_subtasks = ['drd', 'gsk', 'jnk', 'logp', 'qed', 'solubility']

chemcotbench_mol_opt_datasets = []

for subtask in _mol_opt_subtasks:
    reader_cfg = dict(
        input_columns=['prompt'],
        output_column='source_smiles',  # Source for optimization tasks
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
            type='ChemCoTBenchMolOptEvaluator',
            subtask=subtask,
        ),
    )

    chemcotbench_mol_opt_datasets.append(
        dict(
            abbr=f'chemcotbench_mol_opt_{subtask}',
            type='ChemCoTBenchMolOptDataset',
            path='OpenMol/ChemCoTBench',
            subtask=subtask,
            reader_cfg=reader_cfg,
            infer_cfg=infer_cfg,
            eval_cfg=eval_cfg,
        )
    )
