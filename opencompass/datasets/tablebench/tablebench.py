"""
TableBench Dataset evaluator
Reference: https://huggingface.co/datasets/Multilingual-Multimodal-NLP/TableBench
"""

import json
import re
from typing import Any, Dict, List, Optional
import sys
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
             instruction_type: str = 'TCoT',
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
            table_formatted = table
            table_raw = table

            if isinstance(table, dict):
                # Convert structured table to formatted string
                table_formatted = format_table(table)
                table_raw = table

            elif not isinstance(table, str):
                table_formatted = str(table)
                table_raw = {}
            
            # Handle question field
            question = item.get('question', '')
            
            # Handle answer field
            answer = item.get('answer', '')
            
            return {
                'table': table_formatted,
                'table_raw': table_raw,
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
        elif self.metric == 'exact_match_with_final_answer':
            return self._exact_match_score_with_final_answer(predictions, ref_answers)
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

    def _exact_match_score_with_final_answer(self, predictions: List, references: List) -> Dict:
        """Calculate exact match accuracy."""
        correct = 0
        count = 0
        details = []
        
        for pred, ref in zip(predictions, references):
            count += 1
            
            # ⭐ 关键修复：先提取 Final Answer
            pred_extracted = self._extract_final_answer(pred)
            ref_extracted = self._extract_final_answer(ref)
            
            # Normalize strings for comparison
            pred_normalized = self._normalize_answer(pred_extracted)
            ref_normalized = self._normalize_answer(ref_extracted)
            
            is_correct = pred_normalized == ref_normalized
            if is_correct:
                correct += 1
            
            detail = {
                'pred': pred[:200],  # Truncate full response for readability
                'pred_extracted': pred_extracted,  # Show extracted answer
                'answer': ref,
                'correct': is_correct
            }
            details.append(detail)
        
        accuracy = 100 * correct / count if count > 0 else 0
        return {
            'exact_match': accuracy,
            'accuracy': accuracy,
            'correct_count': correct,
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

    
    @staticmethod
    def _extract_final_answer(text: str) -> str:
        """Extract final answer from text with 'Final Answer:' marker."""
        if not isinstance(text, str):
            text = str(text)
        
        # Look for "Final Answer:" marker (case insensitive)
        import re
        match = re.search(r'Final Answer:\s*(.+?)(?:\n|$)', text, re.IGNORECASE | re.DOTALL)
        if match:
            return match.group(1).strip()
        
        # If no marker found, return the whole text
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


@ICL_EVALUATORS.register_module()
class TableBenchNumericalWithPercenteErrorEvaluator(BaseEvaluator):
    """
    Evaluator for numeric answers with percentage-based error tolerance.
    Implements EM_with_error_10 metric (10% relative error tolerance).
    """
    def __init__(self, error_rate: float = 0.10):  
        """
        Args:
            error_rate: Allowed relative error rate (10% = 0.10 by default)
        """
        self.error_rate = error_rate

    def score(self, predictions, references):
        if len(predictions) != len(references):
            return {
                "error": "predictions and references have different length"
            }
        
        # Extract answer strings from references if they are dicts
        ref_answers = []
        for ref in references:
            if isinstance(ref, dict):
                ref_answers.append(ref.get('answer', ''))
            else:
                ref_answers.append(ref)

        correct = 0
        count = 0
        details = []

        for pred, ref in zip(predictions, ref_answers):
            count += 1
            
            pred_extracted = self._extract_final_answer(pred)
            ref_extracted = self._extract_final_answer(ref)
            
            detail = {
                'pred': pred[:200],  # Truncate full response
                'pred_extracted': pred_extracted,  # Show extracted answer
                'answer': ref,
                'correct': False
            }
            
            try:
                pred_num = self._parse_number(pred_extracted)  # 从提取的答案解析
                ref_num = self._parse_number(ref_extracted)
                
                if pred_num is not None and ref_num is not None:
                    # Use percentage-based tolerance
                    if ref_num == 0:
                        # If reference is 0, use absolute comparison
                        if abs(pred_num) < 1e-6:
                            correct += 1
                            detail['correct'] = True
                    else:
                        # Calculate relative error
                        relative_error = abs(pred_num - ref_num) / abs(ref_num)
                        detail['relative_error'] = f"{relative_error:.2%}"
                        
                        if relative_error <= self.error_rate:
                            correct += 1
                            detail['correct'] = True
                else:
                    # Fallback to string comparison
                    if pred_extracted.strip().lower() == ref_extracted.strip().lower():
                        correct += 1
                        detail['correct'] = True
            except Exception as e:
                # If parsing fails, use string comparison
                detail['parse_error'] = str(e)
                if pred_extracted.strip().lower() == ref_extracted.strip().lower():
                    correct += 1
                    detail['correct'] = True
            
            details.append(detail)
        
        accuracy = 100 * correct / count if count > 0 else 0
        return {
            'accuracy': accuracy,
            'EM_with_error_10': accuracy,
            'correct_count': correct,
            'total_count': count,
            'details': details
        }

    @staticmethod
    def _extract_final_answer(text: str) -> str:
        """Extract final answer from text with 'Final Answer:' marker."""
        if not isinstance(text, str):
            text = str(text)
        
        # Look for "Final Answer:" marker (case insensitive)
        import re
        match = re.search(r'Final Answer:\s*(.+?)(?:\n|$)', text, re.IGNORECASE | re.DOTALL)
        if match:
            return match.group(1).strip()
        
        # If no marker found, return the whole text
        return text.strip()

    @staticmethod
    def _parse_number(text):
        """Parse number from text."""
        if isinstance(text, (int, float)):
            return float(text)
        
        if not isinstance(text, str):
            return None
        
        # Remove common number formatting
        text = text.strip().replace(',', '').replace('%', '')
        
        # Try to extract the first number (for multi-value answers like "979000, 3")
        match = re.search(r'-?\d+\.?\d*', text)
        if match:
            try:
                return float(match.group())
            except ValueError:
                return None
        
        return None
    
@ICL_EVALUATORS.register_module()
class TableBenchRougeEvaluator(BaseEvaluator):
    """
    Evaluator using ROUGE-L metric for text similarity,
    Used for open-ended text generation tasks in TableBench.
    """
    def __init__(self):
        """Initialize ROUGE evaluator."""
        try:
            from rouge import Rouge
            self.rouge = Rouge()
        except ImportError:
            raise ImportError("Please install rouge: pip install rouge")

    def score(self, predictions: List, references: List, test_set: Optional[Dataset] = None) -> Dict:
        """Calculate ROUGE-L score."""
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
        
        total_rouge_l = 0
        count = 0
        details = []
        
        for pred, ref in zip(predictions, ref_answers):
            pred_answer = self._extract_final_answer(pred)
            ref_answer = ref.strip()
            
            # Normalize for comparison
            pred_str = pred_answer.lower().strip() if pred_answer else "empty"
            ref_str = ref_answer.lower().strip() if ref_answer else "empty"
            
            if not pred_str or not ref_str:
                rouge_l_score = 0.0
            else:
                try:
                    scores = self.rouge.get_scores(pred_str, ref_str)[0]
                    rouge_l_score = scores['rouge-l']['f'] * 100  # Convert to percentage
                except Exception:
                    rouge_l_score = 0.0
            
            total_rouge_l += rouge_l_score
            count += 1
            
            detail = {
                'pred': pred[:200],  # Truncate full response
                'pred_extracted': pred_answer,  # Show extracted answer
                'answer': ref,
                'rouge_l': rouge_l_score,
                'correct': rouge_l_score > 50  # Consider >50 as acceptable
            }
            details.append(detail)
        
        avg_rouge_l = total_rouge_l / count if count > 0 else 0
        return {
            'ROUGE-L': avg_rouge_l,
            'accuracy': avg_rouge_l,
            'correct_count': sum(1 for d in details if d['correct']),
            'total_count': count,
            'details': details
        }

    @staticmethod
    def _extract_final_answer(text: str) -> str:
        """Extract final answer from text with 'Final Answer:' marker."""
        if not isinstance(text, str):
            text = str(text)
        
        # Look for "Final Answer:" marker (case insensitive)
        import re
        match = re.search(r'Final Answer:\s*(.+?)(?:\n|$)', text, re.IGNORECASE | re.DOTALL)
        if match:
            return match.group(1).strip()
        
        # If no marker found, return the whole text
        return text.strip()
            



        



@ICL_EVALUATORS.register_module()
class TableBenchVisualizationEvaluator(BaseEvaluator):
    """
    Evaluator for visualization code generation in TableBench.
    Evaluates using Pass@1 and ECR@1 (Execution Correctness Rate) metrics.
    Pass@1: Whether the generated code produces correct visualization data (y_preferences match)
    ECR@1: Whether the generated code executes successfully without errors.
    """
    def __init__(self,timeout:int=10):
        """
        Args:
            timeout: Maximum exectuion time is 10 seconds.
        """
        self.timeout = timeout

    def score(self, predictions,references,test_set)->Dict:
        """
        Calculate Pass@1 and ECR@1 metrics.
        Args:
            predictions: List of model predictions
            references: List of ground truth answers or dicts with 'answer' key
            test_set: Optional test dataset
        Returns:
            Dictionary containing evaluation results
        """
        if len(predictions) != len(references):
            return {'error': 'predictions and references have different length'}
    
        ref_data = []
        for ref in references:
            if isinstance(ref, dict):
                ref_data.append(ref.get('answer', ''))
            else:
                ref_data.append(ref)
        
        pass_1_count = 0
        ecr_1_count = 0
        parse_1_count = 0
        total = len(predictions)
        details = []
        
        for idx, (pred, ref, test_item) in enumerate(zip(predictions, ref_data, test_set if test_set else [{}]*total)):
            detail = {'pred': pred[:100], 'answer': ref, 'parse_1': False, 'ecr_1': False, 'pass_1': False}
            
            # 添加调试信息
            print(f"\n=== Sample {idx} ===")
            
            code = self._extract_code(pred)
            if code:
                detail['parse_1'] = True
                parse_1_count += 1
                print(f"✓ Code extracted ({len(code)} chars)")
                
                # 修复：使用 table_raw 获取原始字典格式
                table_data = test_item.get('table_raw', {}) if isinstance(test_item, dict) else {}
                print(f"Table data: {type(table_data)}, has data: {bool(table_data)}")
                
                # 如果 table_raw 不存在，尝试使用原始 table 字段
                if not table_data or isinstance(table_data, str):
                    print("⚠ Warning: table_raw not found or is string, trying to parse table field")
                    # 这是一个临时解决方案，理想情况下应该修改 dataset loader
                
                # 修复：传递 ref (y_references) 作为第 3 个参数
                exec_result = self._execute_code(code, table_data, ref)
                
                print(f"Executed: {exec_result['executed']}")
                print(f"Correct: {exec_result['correct']}")
                if exec_result.get('error'):
                    print(f"Error: {exec_result['error'][:200]}")
                
                if exec_result['executed']:
                    detail['ecr_1'] = True
                    ecr_1_count += 1
                
                # 修复：设置 pass_1 而不是 ecr_1
                if exec_result['correct']:
                    detail['pass_1'] = True
                    pass_1_count += 1
                
                detail['execution_error'] = exec_result.get('error', None)
            else:
                print(f"✗ No code extracted")
            
            details.append(detail)
        
        pass_1_rate = 100 * pass_1_count / total if total > 0 else 0
        ecr_1_rate = 100 * ecr_1_count / total if total > 0 else 0
        parse_1_rate = 100 * parse_1_count / total if total > 0 else 0
        
        print(f"\n=== Summary ===")
        print(f"Parse@1: {parse_1_count}/{total} ({parse_1_rate:.1f}%)")
        print(f"ECR@1: {ecr_1_count}/{total} ({ecr_1_rate:.1f}%)")
        print(f"Pass@1: {pass_1_count}/{total} ({pass_1_rate:.1f}%)")
        
        return {
            'Pass@1': pass_1_rate,
            'ECR@1': ecr_1_rate,
            'Parse@1': parse_1_rate,
            'total_count': total,
            'details': details
        }
    @staticmethod
    def _extract_code(text):
        "Extract Python code from markdown code blocks or raw text"
        if not isinstance(text, str):
            return None
        pattern = r'```python\s*(.*?)\s*```'
        matches = re.findall(pattern, text, re.DOTALL)
        if matches: 
            # use the last code block 
            return matches[-1].strip()
        # If no code block found, check if the entire text looks like code
        if any(keyword in text for keyword in ['import', 'plt', 'numpy', 'pandas','pd','df']):
            return text.strip()
        return None

    def _execute_code(self, code, table_data, y_references):
        """Execute code and check if output matches y_references
        Returns: Dict with keys:
        - executed (bool): Whether code executed without errors
        - correct (bool): Whether output matches y_references
        - error (str): Error message if execution failed
        """
        import tempfile
        import subprocess
        import json
        import pandas as pd
        from pathlib import Path
        import textwrap  # 添加这个导入
        
        result = {'executed': False, 'correct': False, 'error': None}
        try:
            ref_arrays = self._parse_y_references(y_references)
            if ref_arrays is None:
                result['error'] = 'Invalid y_references format'
                return result
            
            # Create temporary directory for execution
            with tempfile.TemporaryDirectory() as tmpdir:
                tmpdir_path = Path(tmpdir)
                # Save table data as csv
                if table_data and 'columns' in table_data and 'data' in table_data:
                    df = pd.DataFrame(table_data['data'], columns=table_data['columns'])
                    csv_path = tmpdir_path / 'table.csv'
                    df.to_csv(csv_path, index=False)
                
                # Prepare execution script - 使用 textwrap.dedent 去除前导空格
                exec_script = textwrap.dedent(f"""
                    import sys
                    import pandas as pd
                    import matplotlib.pyplot as plt
                    import numpy as np
                    import json
                    from pathlib import Path
                    
                    # Suppress matplotlib display
                    plt.ioff()
                    
                    # Change to temp directory
                    import os
                    os.chdir(r'{tmpdir_path}')
                    
                    # Variables to capture data
                    captured_data = []
                    
                    # Override plt methods to capture data
                    original_bar = plt.bar
                    original_plot = plt.plot
                    original_scatter = plt.scatter
                    
                    def capture_bar(*args, **kwargs):
                        if len(args) >= 2:
                            captured_data.append(list(args[1]) if hasattr(args[1], '__iter__') else [args[1]])
                        return original_bar(*args, **kwargs)
                    
                    def capture_plot(*args, **kwargs):
                        for i in range(0, len(args), 2):
                            if i+1 < len(args):
                                y_data = args[i+1]
                                captured_data.append(list(y_data) if hasattr(y_data, '__iter__') else [y_data])
                        return original_plot(*args, **kwargs)
                    
                    def capture_scatter(*args, **kwargs):
                        if len(args) >= 2:
                            captured_data.append(list(args[1]) if hasattr(args[1], '__iter__') else [args[1]])
                        return original_scatter(*args, **kwargs)
                    
                    plt.bar = capture_bar
                    plt.plot = capture_plot
                    plt.scatter = capture_scatter
                    
                    try:
                """).lstrip() + '\n' + '\n'.join('    ' + line for line in code.split('\n')) + textwrap.dedent("""
                        # Save captured data
                        with open('output.json', 'w') as f:
                            json.dump({'data': captured_data, 'success': True}, f)
                    except Exception as e:
                        with open('output.json', 'w') as f:
                            json.dump({'error': str(e), 'success': False}, f)
                """)
                
                script_path = tmpdir_path / 'exec_script.py'
                script_path.write_text(exec_script)
                
                # Execute with timeout
                try:
                    proc = subprocess.run(
                        [sys.executable, str(script_path)],
                        timeout=self.timeout,
                        capture_output=True,
                        text=True,
                        cwd=tmpdir
                    )
                    
                    # Read output
                    output_path = tmpdir_path / 'output.json'
                    if output_path.exists():
                        with open(output_path, 'r') as f:
                            output = json.load(f)
                        
                        if output.get('success'):
                            result['executed'] = True
                            
                            # Compare captured data with y_references
                            captured = output.get('data', [])
                            if self._compare_data_arrays(captured, ref_arrays):
                                result['correct'] = True
                        else:
                            result['error'] = output.get('error', 'Unknown error')
                    else:
                        result['error'] = f"Execution failed: {proc.stderr}"
                        
                except subprocess.TimeoutExpired:
                    result['error'] = f'Execution timeout (>{self.timeout}s)'
                except Exception as e:
                    result['error'] = f'Execution error: {str(e)}'
                    
        except Exception as e:
            result['error'] = f'Setup error: {str(e)}'
        
        return result

    @staticmethod
    def _parse_y_references(y_ref):
        """Parse y references string into list of lists."""
        if isinstance(y_ref, list):
            return y_ref

        if isinstance(y_ref, str):
            import ast
            try:
                # 如果包含 'y_references ='，提取等号后的所有内容
                if 'y_references' in y_ref.lower():
                    # 找到等号位置
                    eq_pos = y_ref.find('=')
                    if eq_pos != -1:
                        # 提取等号后的所有内容并去除首尾空白
                        y_ref = y_ref[eq_pos + 1:].strip()
                
                # 使用 ast.literal_eval 安全解析 Python 字面量
                parsed = ast.literal_eval(y_ref)
                if isinstance(parsed, list):
                    return parsed
            except Exception as e:
                print(f"⚠ Warning: Failed to parse y_references: {str(e)[:150]}")
                # 尝试查看原始内容
                print(f"  Raw content (first 200 chars): {y_ref[:200]}")
                pass

        return None

    @staticmethod
    def _compare_data_arrays(captured, reference, tolerance: float = 1e-2):
        """Compare captured data arrays with reference arrays"""
        if not captured or not reference:
            return False
        
        #Must have same number or data series
        if len(captured) != len(reference):
            return False
        
        #Compare each data series
        for cap_series, ref_series in zip(captured, reference):
            if not isinstance(cap_series, (list, tuple)) or not isinstance(ref_series, (list, tuple)):
                continue

            #Must have same length
            if len(cap_series) != len(ref_series):
                return False
            
            #Compare value with tolerance
            for cap_val, ref_val in zip(cap_series, ref_series):
                try:
                    cap_num = float(cap_val)
                    ref_num = float(ref_val)
                    if abs(cap_num - ref_num) > tolerance:
                        return False
                except (ValueError, TypeError):
                    # if not numeric, do string comparison
                    if str(cap_val) != str(ref_val):
                        return False
        return True


       
