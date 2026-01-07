#!/usr/bin/env python3
"""动态生成OpenCompass配置文件"""

import os
from pathlib import Path

# Benchmark配置
BENCHMARKS = [
    # (category, task_name, dataset_var, config_module)
    ('panorama', 'par4pc_cot', 'panorama_par4pc_cot_datasets', 'panorama.panorama_par4pc_cot_gen'),
    ('panorama', 'pi4pc_cot', 'panorama_pi4pc_cot_datasets', 'panorama.panorama_pi4pc_cot_gen'),
    ('panorama', 'noc4pc_cot', 'panorama_noc4pc_cot_datasets', 'panorama.panorama_noc4pc_cot_gen'),
    ('chemcotbench', 'mol_und', 'chemcotbench_mol_und_datasets', 'chemcotbench.chemcotbench_mol_und_gen'),
    ('chemcotbench', 'mol_edit', 'chemcotbench_mol_edit_datasets', 'chemcotbench.chemcotbench_mol_edit_gen'),
    ('chemcotbench', 'mol_opt', 'chemcotbench_mol_opt_datasets', 'chemcotbench.chemcotbench_mol_opt_gen'),
    ('chemcotbench', 'reaction', 'chemcotbench_reaction_datasets', 'chemcotbench.chemcotbench_reaction_gen'),
    ('tablebench', 'data_analysis', 'tablebench_data_analysis_datasets', 'tablebench.tablebench_data_analysis_gen'),
    ('tablebench', 'fact_checking', 'tablebench_fact_checking_datasets', 'tablebench.tablebench_fact_checking_gen'),
    ('tablebench', 'numerical_reasoning', 'tablebench_numerical_datasets', 'tablebench.tablebench_numerical_reasoning_gen'),
    ('tablebench', 'visualization', 'tablebench_visualization_datasets', 'tablebench.tablebench_visualization_gen'),
    ('bioprobench', 'gen', 'bioprobench_gen_datasets', 'bioprobench.bioprobench_gen'),
    ('bioprobench', 'ord', 'bioprobench_ord_datasets', 'bioprobench.bioprobench_ord'),
    ('bioprobench', 'err', 'bioprobench_err_datasets', 'bioprobench.bioprobench_err'),
    ('bioprobench', 'pqa', 'bioprobench_pqa_datasets', 'bioprobench.bioprobench_pqa'),
]

# 配置模板
CONFIG_TEMPLATE = '''"""Auto-generated config for {category}/{task_name}"""
from mmengine.config import read_base
from opencompass.models import OpenAI
from opencompass.partitioners import NaivePartitioner
from opencompass.runners import LocalRunner
from opencompass.tasks import OpenICLInferTask, OpenICLEvalTask

with read_base():
    from opencompass.configs.datasets.{config_module} import {dataset_var}

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
datasets = {dataset_var}
for dataset in datasets:
    if 'reader_cfg' not in dataset:
        dataset['reader_cfg'] = {{}}
    dataset['reader_cfg']['test_range'] = '[0:10]'

    # 同时修改eval_cfg中dataset_cfg的reader_cfg（用于LLM Judge评估器）
    if 'eval_cfg' in dataset and 'evaluator' in dataset['eval_cfg']:
        evaluator = dataset['eval_cfg']['evaluator']
        if 'dataset_cfg' in evaluator:
            if 'reader_cfg' not in evaluator['dataset_cfg']:
                evaluator['dataset_cfg']['reader_cfg'] = {{}}
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

work_dir = './outputs/{category}/{task_name}'
'''


def generate_configs():
    """生成所有配置文件"""
    base_dir = Path(__file__).parent

    for category, task_name, dataset_var, config_module in BENCHMARKS:
        # 创建目录
        task_dir = base_dir / category / task_name
        task_dir.mkdir(parents=True, exist_ok=True)

        # 生成配置文件
        config_content = CONFIG_TEMPLATE.format(
            category=category,
            task_name=task_name,
            dataset_var=dataset_var,
            config_module=config_module,
        )

        config_path = task_dir / 'config.py'
        config_path.write_text(config_content)
        print(f'Generated: {config_path}')


if __name__ == '__main__':
    generate_configs()
    print('All configs generated!')
