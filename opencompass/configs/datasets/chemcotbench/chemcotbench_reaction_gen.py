"""ChemCoTBench Reaction Zero-shot Configuration.

Evaluates:
- fs: Forward synthesis prediction
- retro: Retrosynthesis prediction
- rcr: Reaction condition recommendation
- nepp: Next elementary step product prediction
- mechsel: Mechanism route selection
"""

from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer

# Dataset configuration
_reaction_subtasks = ['fs', 'retro', 'rcr', 'nepp', 'mechsel']

chemcotbench_reaction_datasets = []

for subtask in _reaction_subtasks:
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
            type='ChemCoTBenchReactionEvaluator',
            subtask=subtask,
        ),
    )

    chemcotbench_reaction_datasets.append(
        dict(
            abbr=f'chemcotbench_reaction_{subtask}',
            type='ChemCoTBenchReactionDataset',
            path='OpenMol/ChemCoTBench',
            subtask=subtask,
            reader_cfg=reader_cfg,
            infer_cfg=infer_cfg,
            eval_cfg=eval_cfg,
        )
    )
