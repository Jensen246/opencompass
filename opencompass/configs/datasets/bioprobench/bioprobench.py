from mmengine.config import read_base

with read_base():
    from .bioprobench_gen import bioprobench_gen_datasets
    from .bioprobench_pqa import bioprobench_pqa_datasets
    from .bioprobench_err import bioprobench_err_datasets
    from .bioprobench_ord import bioprobench_ord_datasets

datasets = (
    bioprobench_gen_datasets
    + bioprobench_pqa_datasets
    + bioprobench_err_datasets
    + bioprobench_ord_datasets
)