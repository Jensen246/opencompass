"""Auto-generated config for tablebench/numerical_reasoning"""
from mmengine.config import read_base
from opencompass.models import OpenAI
from opencompass.partitioners import NaivePartitioner
from opencompass.runners import LocalRunner
from opencompass.tasks import OpenICLInferTask, OpenICLEvalTask

with read_base():
    from opencompass.configs.datasets.tablebench.tablebench_numerical_reasoning_gen import tablebench_numerical_datasets

# 模型配置
api_meta_template = dict(
    round=[
        dict(role='HUMAN', api_role='HUMAN'),
        dict(role='BOT', api_role='BOT', generate=True),
    ],
)

models = [
    dict(
        abbr='gpt-4o-mini',
        type=OpenAI,
        path='gpt-4o-mini',
        openai_api_base='http://localhost:3000/v1/chat/completions',
        key='sk-1234',
        meta_template=api_meta_template,
        query_per_second=5,
        max_out_len=4096,
        max_seq_len=128000,
        batch_size=8,
    )
]

# 数据集配置（限制10个样例）
datasets = tablebench_numerical_datasets
for dataset in datasets:
    if 'reader_cfg' not in dataset:
        dataset['reader_cfg'] = {}
    dataset['reader_cfg']['test_range'] = '[0:10]'

    # 同时修改eval_cfg中dataset_cfg的reader_cfg（用于LLM Judge评估器）
    if 'eval_cfg' in dataset and 'evaluator' in dataset['eval_cfg']:
        evaluator = dataset['eval_cfg']['evaluator']
        if 'dataset_cfg' in evaluator:
            if 'reader_cfg' not in evaluator['dataset_cfg']:
                evaluator['dataset_cfg']['reader_cfg'] = {}
            evaluator['dataset_cfg']['reader_cfg']['test_range'] = '[0:10]'

# 执行配置
infer = dict(
    partitioner=dict(type=NaivePartitioner),
    runner=dict(
        type=LocalRunner,
        max_num_workers=1000,
        task=dict(type=OpenICLInferTask),
    ),
)

eval = dict(
    partitioner=dict(type=NaivePartitioner),
    runner=dict(
        type=LocalRunner,
        max_num_workers=4,
        task=dict(type=OpenICLEvalTask),
    ),
)

work_dir = './outputs/tablebench/numerical_reasoning'
