"""TableBench完整评测配置 - 包含所有任务类型"""
from mmengine.config import read_base

with read_base():
    from .tablebench_fact_checking_gen import tablebench_fact_checking_datasets
    from .tablebench_numerical_reasoning_gen import tablebench_numerical_datasets
    from .tablebench_data_analysis_gen import tablebench_data_analysis_datasets
    from .tablebench_visualization_gen import tablebench_visualization_datasets

# 合并所有数据集
tablebench_datasets = (
    tablebench_fact_checking_datasets +
    tablebench_numerical_datasets +
    tablebench_data_analysis_datasets +
    tablebench_visualization_datasets
)



 