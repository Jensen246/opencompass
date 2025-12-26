from mmengine.config import read_base

with read_base():
    from opencompass.configs.models.qwen3.lmdeploy_qwen3_0_6b import models
    from opencompass.configs.datasets.bioprobench.bioprobench_pqa import bioprobench_pqa_datasets