import json
import re

from datasets import load_dataset

from opencompass.openicl.icl_evaluator import BaseEvaluator
from opencompass.registry import ICL_EVALUATORS, LOAD_DATASET

from .base import BaseDataset


@LOAD_DATASET.register_module()
class NvBench2Dataset(BaseDataset):
    """nvBench 2.0 数据集，遵循官方仓库格式"""

    @staticmethod
    def load(path: str = 'TianqiLuo/nvBench2.0', split: str = 'test', **kwargs):
        dataset = load_dataset(path, split=split)

        def process_item(item):
            schema = json.loads(item['table_schema'])
            item['data_schema'] = str(schema.get('table_columns', []))
            item['value_example'] = str(schema.get('column_examples', {}))
            item['unique_value_num'] = str(schema.get('unique_value_counts', {}))
            return item

        dataset = dataset.map(process_item)
        return dataset


@ICL_EVALUATORS.register_module()
class NvBench2Evaluator(BaseEvaluator):
    """nvBench 2.0 评估器，遵循官方 evaluation.py 实现"""

    # mark 类型映射（模型输出 -> gold answer 格式）
    # Gold answer 使用: pie, rect, point, bar, line, boxplot
    # Prompt 指导模型输出: arc (=pie)
    MARK_MAPPING = {'arc': 'pie'}

    def __init__(self, k_values=None):
        if k_values is None:
            k_values = [1, 3, 5]
        self.k_values = k_values

    def normalize_chart_order(self, chart):
        """规范化 Vega-Lite 图表对象，只保留评估必需的核心属性"""
        if not isinstance(chart, dict):
            return chart

        normalized = {}

        # 1. mark（应用 arc->pie 映射）
        if 'mark' in chart:
            mark = chart['mark']
            normalized['mark'] = self.MARK_MAPPING.get(mark, mark)

        # 2. encoding（只保留核心属性：field, aggregate, bin, sort）
        if 'encoding' in chart and isinstance(chart['encoding'], dict):
            normalized['encoding'] = {}
            core_props = ['field', 'aggregate', 'bin', 'sort']

            for channel, channel_data in chart['encoding'].items():
                if isinstance(channel_data, dict):
                    norm_channel = {}
                    for prop in core_props:
                        if prop in channel_data:
                            norm_channel[prop] = channel_data[prop]
                    if norm_channel:
                        normalized['encoding'][channel] = norm_channel

        # 3. transform（只保留有效的 filter，不添加空列表）
        if 'transform' in chart and isinstance(chart['transform'], list):
            valid_transforms = []
            for t in chart['transform']:
                if isinstance(t, dict) and 'filter' in t and t['filter']:
                    valid_transforms.append(t)
            if valid_transforms:
                normalized['transform'] = valid_transforms

        return normalized

    def deep_compare_charts(self, chart1, chart2):
        """深度比较两个可视化规范，忽略键/列表顺序"""
        if type(chart1) != type(chart2):
            return False
        if isinstance(chart1, dict):
            if len(chart1) != len(chart2):
                return False
            for key in chart1:
                if key not in chart2:
                    return False
                if not self.deep_compare_charts(chart1[key], chart2[key]):
                    return False
            return True
        elif isinstance(chart1, list):
            if len(chart1) != len(chart2):
                return False
            matched = [False] * len(chart2)
            for item1 in chart1:
                found = False
                for i, item2 in enumerate(chart2):
                    if not matched[i] and self.deep_compare_charts(item1, item2):
                        matched[i] = True
                        found = True
                        break
                if not found:
                    return False
            return True
        else:
            return chart1 == chart2

    def parse_predictions(self, pred_str):
        """从模型输出中解析可视化列表"""
        if not isinstance(pred_str, str):
            pred_str = str(pred_str)

        # 尝试解析完整 JSON（包含 final_output）
        try:
            # 去除 markdown 代码块标记
            clean_str = pred_str.strip()
            if clean_str.startswith('```json'):
                clean_str = clean_str[7:]
            if clean_str.startswith('```'):
                clean_str = clean_str[3:]
            if clean_str.endswith('```'):
                clean_str = clean_str[:-3]
            clean_str = clean_str.strip()

            obj = json.loads(clean_str)
            if isinstance(obj, dict) and 'final_output' in obj:
                return obj['final_output']
            if isinstance(obj, list):
                return obj
        except json.JSONDecodeError:
            pass

        # 格式1: "final_output": [...] (Step prompt)
        match = re.search(r'"final_output"\s*:\s*(\[[\s\S]*?\])\s*\}', pred_str)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                pass

        # 格式2: #OUTPUT:[...] (Basic prompt)
        match = re.search(r'#OUTPUT:\s*(\[[\s\S]*\])', pred_str)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                pass

        # 格式3: 直接 JSON 数组
        match = re.search(r'\[\s*\{[\s\S]*\}\s*\]', pred_str)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass

        return []

    def score(self, predictions, references):
        """计算 Hit@K, P@K, R@K, F1@K 指标"""
        results = {}
        for k in self.k_values:
            results[f'Hit@{k}'] = 0.0
            results[f'P@{k}'] = 0.0
            results[f'R@{k}'] = 0.0
            results[f'F1@{k}'] = 0.0

        valid_count = 0
        for pred_str, ref_str in zip(predictions, references):
            preds = self.parse_predictions(pred_str)
            if isinstance(ref_str, str):
                try:
                    refs = json.loads(ref_str)
                except json.JSONDecodeError:
                    refs = []
            else:
                refs = ref_str if isinstance(ref_str, list) else []

            if not refs:
                continue
            valid_count += 1

            # 标准化图表结构
            preds = [self.normalize_chart_order(p) for p in preds]
            refs = [self.normalize_chart_order(r) for r in refs]

            for k in self.k_values:
                preds_k = preds[:k]
                if not preds_k:
                    continue

                # 贪婪二分匹配
                matched = 0
                ref_matched = [False] * len(refs)
                for p in preds_k:
                    for i, r in enumerate(refs):
                        if not ref_matched[i] and self.deep_compare_charts(p, r):
                            ref_matched[i] = True
                            matched += 1
                            break

                hit = 1.0 if matched > 0 else 0.0
                precision = matched / len(preds_k)
                recall = matched / len(refs)
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

                results[f'Hit@{k}'] += hit
                results[f'P@{k}'] += precision
                results[f'R@{k}'] += recall
                results[f'F1@{k}'] += f1

        if valid_count > 0:
            for key in results:
                # 转换为百分数，保留两位小数（与论文格式一致）
                results[key] = round(results[key] / valid_count * 100, 2)

        return results
