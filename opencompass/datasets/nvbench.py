from datasets import load_dataset

from opencompass.registry import LOAD_DATASET

from .base import BaseDataset


@LOAD_DATASET.register_module()
class NvBench2Dataset(BaseDataset):

    @staticmethod
    def load(path: str = 'TianqiLuo/nvBench2.0', split: str = 'test', **kwargs):
        dataset = load_dataset(path, split=split)
        return dataset
