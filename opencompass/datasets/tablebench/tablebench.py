"""
TableBench Dataset evaluator
Reference: https://huggingface.co/datasets/Multilingual-Multimodal-NLP/TableBench
"""

import json
import re
from typing import Any, Dict, List, Optional

from datasets import Dataset, load_dataset

from opencompass.openicl.icl_evaluator import BaseEvaluator
from opencompass.registry import ICL_EVALUATORS, LOAD_DATASET

from ..base import BaseDataset


def format_table(table_data: Any) -> str:
    """
    Format table data into a readable string representation.
    
    Args:
        table_data: Table data (dict with 'columns' and 'data' keys, or other formats)
        
    Returns:
        Formatted table string
    """
    if not table_data:
        return ""
    
    # If already a string, return as is
    if isinstance(table_data, str):
        return table_data
    
    # TableBench format: dict with 'columns' and 'data' keys
    if isinstance(table_data, dict) and 'columns' in table_data and 'data' in table_data:
        columns = table_data['columns']
        data = table_data['data']
        
        if not columns or not data:
            return str(table_data)
        
        # Create header row
        lines = []
        header_line = " | ".join(str(col) for col in columns)
        lines.append(header_line)
        lines.append("-" * len(header_line))
        
        # Add data rows
        for row in data:
            row_line = " | ".join(str(cell) for cell in row)
            lines.append(row_line)
        
        return "\n".join(lines)
    
    # If it's a list of dicts (alternative table format)
    if isinstance(table_data, list) and len(table_data) > 0:
        if isinstance(table_data[0], dict):
            # Get headers from first row
            headers = list(table_data[0].keys())
            
            # Create header row
            lines = []
            header_line = " | ".join(str(h) for h in headers)
            lines.append(header_line)
            lines.append("-" * len(header_line))
            
            # Add data rows
            for row in table_data:
                row_line = " | ".join(str(row.get(h, '')) for h in headers)
                lines.append(row_line)
            
            return "\n".join(lines)
    
    # If it's a single dict without columns/data structure
    if isinstance(table_data, dict):
        lines = []
        header_line = " | ".join(str(k) for k in table_data.keys())
        lines.append(header_line)
        lines.append("-" * len(header_line))
        value_line = " | ".join(str(v) for v in table_data.values())
        lines.append(value_line)
        return "\n".join(lines)
    
    # Fallback: convert to string
    return str(table_data)


@LOAD_DATASET.register_module()
class TableBenchDataset(BaseDataset):
    """
    TableBench Dataset Loader - loads from HuggingFace
    
    Args:
        path: HuggingFace dataset path (default: 'Multilingual-Multimodal-NLP/TableBench')
        qtype: Question type filter (e.g., 'DataAnalysis', 'NumericalReasoning', 'FactChecking', 'Visualization')
        qsubtype: Question subtype filter (e.g., 'StatisticalAnalysis', 'Aggregation', etc.)
        instruction_type: Instruction type (DP, TCoT, SCoT, PoT) - loads specific data file
        split: Dataset split (default: 'test')
    """

    @staticmethod
    def load(path: str = 'Multilingual-Multimodal-NLP/TableBench',
             qtype: Optional[str] = None,
             qsubtype: Optional[str] = None,
             instruction_type: str = 'DP',
             **kwargs) -> Dataset:
        """
        Load TableBench dataset from HuggingFace.
        
        Args:
            path: HuggingFace dataset path
            qtype: Filter by question type (DataAnalysis, NumericalReasoning, FactChecking, Visualization)
            qsubtype: Filter by question subtype (more specific task type)
            instruction_type: Type of instruction (DP, TCoT, SCoT, PoT) - determines which data file to load
            split: Dataset split to load (default: 'test')
            
        Returns:
            Processed Dataset with columns: table, question, answer, qtype, qsubtype
        """
        # Ensure we use the correct HuggingFace path
        if not path or '/' not in path or path.startswith('./') or path.startswith('../'):
            path = 'Multilingual-Multimodal-NLP/TableBench'
        
        try:
            # Load dataset with specific data file to avoid column mismatch
            # Use instruction_type to specify which file to load
            data_file = f"TableBench_{instruction_type}.jsonl"
            
            # Try loading with specific data file
            ds = load_dataset(
                path, 
                data_files=data_file,
                split='train'
            )
        
            # Filter by qtype if specified
            if qtype:
                ds = ds.filter(lambda x: x.get('qtype') == qtype)
        
            # Filter by qsubtype if specified
            if qsubtype:
                ds = ds.filter(lambda x: x.get('qsubtype') == qsubtype)
                            
        except Exception as e:
            raise ValueError(
                f"Failed to load dataset from HuggingFace path '{path}' "
                f"with instruction_type '{instruction_type}' and split '{split}'. Error: {str(e)}\n"
                f"Please check:\n"
                f"1. The dataset exists at: https://huggingface.co/datasets/{path}\n"
                f"2. You have internet connection or the dataset is cached\n"
                f"3. The instruction_type is correct (DP, TCoT, SCoT, or PoT)\n"
                f"4. Try clearing the HuggingFace cache if issues persist"
            )
        
        def process_item(item: Dict[str, Any]) -> Dict[str, Any]:
            """Process each dataset item."""
            # Handle table field (should be dict with 'columns' and 'data')
            table = item.get('table', '')
            if isinstance(table, dict):
                # Convert structured table to formatted string
                table = format_table(table)
            elif not isinstance(table, str):
                table = str(table)
            
            # Handle question field
            question = item.get('question', '')
            
            # Handle answer field
            answer = item.get('answer', '')
            
            return {
                'table': table,
                'question': str(question),
                'answer': str(answer),
                'qtype': item.get('qtype', ''),
                'qsubtype': item.get('qsubtype', ''),
                # Preserve additional fields
                'id': item.get('id', ''),
                'instruction': item.get('instruction', ''),
                'instruction_type': item.get('instruction_type', instruction_type),
                'chart_type': item.get('chart_type', ''),
            }
        
        processed_ds = ds.map(process_item)
        return processed_ds


@ICL_EVALUATORS.register_module()
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

    def score(self, predictions: List, references: List, test_set: Optional[Dataset] = None) -> Dict:
        """
        Calculate evaluation metrics.
        
        Args:
            predictions: List of model predictions
            references: List of ground truth answers or dicts with 'answer' key
            test_set: Optional test dataset
            
        Returns:
            Dictionary containing evaluation results
        """
        if len(predictions) != len(references):
            return {
                'error': 'predictions and references have different length'
            }
        
        # Extract answer strings from references if they are dicts
        ref_answers = []
        for ref in references:
            if isinstance(ref, dict):
                ref_answers.append(str(ref.get('answer', '')))
            else:
                ref_answers.append(str(ref))
        
        if self.metric == 'exact_match':
            return self._exact_match_score(predictions, ref_answers)
        elif self.metric == 'f1':
            return self._f1_score(predictions, ref_answers)
        elif self.metric == 'accuracy':
            return self._accuracy_score(predictions, ref_answers)
        else:
            return self._exact_match_score(predictions, ref_answers)

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
            'total_count': count,
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
            'total_count': count,
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
        text = re.sub(r'[^\w\s]', ' ', text)
        text = ' '.join(text.split())
        
        return text.strip()


@ICL_EVALUATORS.register_module()
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

    def score(self, predictions: List, references: List, test_set: Optional[Dataset] = None) -> Dict:
        """Calculate accuracy for numeric answers."""
        if len(predictions) != len(references):
            return {
                'error': 'predictions and references have different length'
            }
        
        # Extract answer strings from references if they are dicts
        ref_answers = []
        for ref in references:
            if isinstance(ref, dict):
                ref_answers.append(str(ref.get('answer', '')))
            else:
                ref_answers.append(str(ref))
        
        correct = 0
        count = 0
        details = []
        
        for pred, ref in zip(predictions, ref_answers):
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
            'total_count': count,
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
        text = text.strip().replace(',', '').replace('%', '')
        
        # Try to extract number
        match = re.search(r'-?\d+\.?\d*', text)
        if match:
            try:
                return float(match.group())
            except ValueError:
                return None
        
        return None