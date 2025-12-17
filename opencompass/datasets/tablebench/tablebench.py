"""
TableBench Dataset evaluator
Reference: https://github.com/TableBench/TableBench
"""

import json
import os
from typing import Dict, List

from datasets import Dataset

from opencompass.openicl import BaseEvaluator
from opencompass.registry import LOAD_DATASET
from opencompass.utils import get_data_path

from ..base import BaseDataset

@LOAD_DATASET.register_module()
class TableBenchDataset(BaseDataset):
    """
    TableBench Dataset Loader
    
    Args:
        path: Path to the dataset directory
        task_type: Type of task (e.g., 'table_qa', 'fact_verification', 'table_to_text')
        subset: Optional subset name
    """

    @staticmethod
    def load(path: str, task_type: str = 'TQA_test', subset: str = None, **kwargs):
        """
        Load TableBench dataset from local files.
        
        The expected data format is JSONL with fields:
        - table: The table data (list of dicts or string)
        - question: The question or prompt
        - answer: Ground truth answer
        - metadata: Optional metadata
        """
        path = get_data_path(path, local_mode=True)
        
        # Construct file path based on task_type and subset
        if subset:
            file_path = os.path.join(path, task_type, f'{subset}.jsonl')
        else:
            file_path = os.path.join(path, f'{task_type}.jsonl')
        
        data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    
                    # Preprocess table data
                    table = item.get('table', '')
                    if isinstance(table, list):
                        # Convert list of dicts to formatted string
                        table = format_table(table)
                    
                    processed_item = {
                        'table': table,
                        'question': item.get('question', ''),
                        'answer': item.get('answer', ''),
                        'task_type': task_type,
                    }
                    
                    # Add optional fields
                    if 'metadata' in item:
                        processed_item['metadata'] = item['metadata']
                    if 'table_id' in item:
                        processed_item['table_id'] = item['table_id']
                    
                    data.append(processed_item)
        
        dataset = Dataset.from_list(data)
        return dataset


def format_table(table_data: List[Dict]) -> str:
    """
    Format table data (list of dicts) into a readable string representation.
    
    Args:
        table_data: List of dictionaries representing table rows
        
    Returns:
        Formatted table string
    """
    if not table_data:
        return ""
    
    # Get headers from first row
    headers = list(table_data[0].keys())
    
    # Create header row
    lines = []
    header_line = " | ".join(headers)
    lines.append(header_line)
    lines.append("-" * len(header_line))
    
    # Add data rows
    for row in table_data:
        row_line = " | ".join(str(row.get(h, '')) for h in headers)
        lines.append(row_line)
    
    return "\n".join(lines)


class TableBenchEvaluator(BaseEvaluator):
    """
    TableBench Evaluator
    
    Evaluates model predictions against ground truth answers.
    Supports multiple evaluation metrics based on task type.
    """

    def __init__(self, metric: str = 'exact_match'):
        """
        Args:
            metric: Evaluation metric to use ('exact_match', 'f1', 'accuracy')
        """
        self.metric = metric

    def score(self, predictions: List, references: List) -> Dict:
        """
        Calculate evaluation metrics.
        
        Args:
            predictions: List of model predictions
            references: List of ground truth answers
            
        Returns:
            Dictionary containing evaluation results
        """
        if len(predictions) != len(references):
            return {
                'error': 'predictions and references have different length'
            }
        
        if self.metric == 'exact_match':
            return self._exact_match_score(predictions, references)
        elif self.metric == 'f1':
            return self._f1_score(predictions, references)
        elif self.metric == 'accuracy':
            return self._accuracy_score(predictions, references)
        else:
            return self._exact_match_score(predictions, references)

    def _exact_match_score(self, predictions: List, references: List) -> Dict:
        """Calculate exact match accuracy."""
        correct = 0
        count = 0
        details = []
        
        for pred, ref in zip(predictions, references):
            detail = {'pred': pred, 'answer': ref, 'correct': False}
            count += 1
            
            # Normalize strings for comparison
            pred_normalized = self._normalize_answer(pred)
            ref_normalized = self._normalize_answer(ref)
            
            if pred_normalized == ref_normalized:
                correct += 1
                detail['correct'] = True
            
            details.append(detail)
        
        accuracy = 100 * correct / count if count > 0 else 0
        return {
            'exact_match': accuracy,
            'accuracy': accuracy,
            'details': details
        }

    def _accuracy_score(self, predictions: List, references: List) -> Dict:
        """Calculate accuracy (alias for exact match)."""
        return self._exact_match_score(predictions, references)

    def _f1_score(self, predictions: List, references: List) -> Dict:
        """Calculate token-level F1 score."""
        total_f1 = 0
        count = 0
        details = []
        
        for pred, ref in zip(predictions, references):
            pred_tokens = self._normalize_answer(pred).split()
            ref_tokens = self._normalize_answer(ref).split()
            
            common_tokens = set(pred_tokens) & set(ref_tokens)
            
            if len(pred_tokens) == 0 or len(ref_tokens) == 0:
                f1 = 0.0
            else:
                precision = len(common_tokens) / len(pred_tokens) if pred_tokens else 0
                recall = len(common_tokens) / len(ref_tokens) if ref_tokens else 0
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            total_f1 += f1
            count += 1
            
            detail = {
                'pred': pred,
                'answer': ref,
                'f1': f1,
                'correct': f1 > 0.8  # Consider F1 > 0.8 as correct
            }
            details.append(detail)
        
        avg_f1 = 100 * total_f1 / count if count > 0 else 0
        return {
            'f1_score': avg_f1,
            'accuracy': avg_f1,
            'details': details
        }

    @staticmethod
    def _normalize_answer(text: str) -> str:
        """Normalize answer text for comparison."""
        if not isinstance(text, str):
            text = str(text)
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove punctuation and extra whitespace
        import re
        text = re.sub(r'[^\w\s]', ' ', text)
        text = ' '.join(text.split())
        
        return text.strip()


class TableBenchNumericEvaluator(BaseEvaluator):
    """
    Evaluator for numeric answers in TableBench.
    Handles floating point comparisons with tolerance.
    """

    def __init__(self, tolerance: float = 1e-3):
        """
        Args:
            tolerance: Tolerance for numeric comparison
        """
        self.tolerance = tolerance

    def score(self, predictions: List, references: List) -> Dict:
        """Calculate accuracy for numeric answers."""
        if len(predictions) != len(references):
            return {
                'error': 'predictions and references have different length'
            }
        
        correct = 0
        count = 0
        details = []
        
        for pred, ref in zip(predictions, references):
            detail = {'pred': pred, 'answer': ref, 'correct': False}
            count += 1
            
            try:
                # Try to parse as numbers
                pred_num = self._parse_number(pred)
                ref_num = self._parse_number(ref)
                
                if pred_num is not None and ref_num is not None:
                    if abs(pred_num - ref_num) < self.tolerance:
                        correct += 1
                        detail['correct'] = True
                else:
                    # Fallback to string comparison
                    if str(pred).strip().lower() == str(ref).strip().lower():
                        correct += 1
                        detail['correct'] = True
            except Exception:
                # If parsing fails, use string comparison
                if str(pred).strip().lower() == str(ref).strip().lower():
                    correct += 1
                    detail['correct'] = True
            
            details.append(detail)
        
        accuracy = 100 * correct / count if count > 0 else 0
        return {
            'accuracy': accuracy,
            'details': details
        }

    @staticmethod
    def _parse_number(text):
        """Parse number from text."""
        if isinstance(text, (int, float)):
            return float(text)
        
        if not isinstance(text, str):
            return None
        
        # Remove common number formatting
        import re
        text = text.strip().replace(',', '').replace('%', '')
        
        # Try to extract number
        match = re.search(r'-?\d+\.?\d*', text)
        if match:
            try:
                return float(match.group())
            except ValueError:
                return None
        
        return None
