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
from rouge import Rouge


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
    
    Dataset files available on HuggingFace:
    - TableBench_DP.jsonl (Direct Prompting)
    - TableBench_PoT.jsonl (Program of Thought)
    - TableBench_SCoT.jsonl (Symbolic Chain of Thought)  
    - TableBench_TCoT.jsonl (Textual Chain of Thought)
    
    Args:
        path: HuggingFace dataset path (default: 'Multilingual-Multimodal-NLP/TableBench')
        qtype: Question type filter (e.g., 'DataAnalysis', 'NumericalReasoning', 'FactChecking', 'Visualization')
        qsubtype: Question subtype filter (e.g., 'StatisticalAnalysis', 'Aggregation', etc.)
        instruction_type: Instruction type (DP, TCoT, SCoT, PoT) - loads specific data file
                         Should be set in config files. Default 'DP' is used if not specified.
    """

    @staticmethod
    def load(path: str = 'Multilingual-Multimodal-NLP/TableBench',
             qtype: Optional[str] = None,
             qsubtype: Optional[str] = None,
             instruction_type: str = None,  # ✅ 改为 'DP' 作为默认值，与配置一致
             **kwargs) -> Dataset:
        """
        Load TableBench dataset from HuggingFace.
        
        Args:
            path: HuggingFace dataset path
            qtype: Filter by question type (DataAnalysis, NumericalReasoning, FactChecking, Visualization)
            qsubtype: Filter by question subtype (more specific task type)
            instruction_type: Type of instruction (DP, TCoT, SCoT, PoT) - determines which data file to load
                            Default: 'DP' (Direct Prompting)
            
        Returns:
            Processed Dataset with columns: table, question, answer, qtype, qsubtype
        """
        # Ensure we use the correct HuggingFace path
        if not path or '/' not in path or path.startswith('./') or path.startswith('../'):
            path = 'Multilingual-Multimodal-NLP/TableBench'
        
        try:
            if instruction_type is None:
                data_file = 'TableBench.jsonl'
            # Load dataset with specific data file to avoid column mismatch
            # Files only have 'train' split, not 'test'
            else:
                data_file = f"TableBench_{instruction_type}.jsonl"
            
            ds = load_dataset(
                path, 
                data_files=data_file,
                split='train'  # Important: only 'train' split exists
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
                f"with instruction_type '{instruction_type}'. Error: {str(e)}\n"
                f"Please check:\n"
                f"1. The dataset exists at: https://huggingface.co/datasets/{path}\n"
                f"2. You have internet connection or the dataset is cached\n"
                f"3. The instruction_type is correct (DP, TCoT, SCoT, or PoT)\n"
                f"4. Available files: TableBench_DP.jsonl, TableBench_PoT.jsonl, "
                f"TableBench_SCoT.jsonl, TableBench_TCoT.jsonl"
            )
        
        def process_item(item: Dict[str, Any]) -> Dict[str, Any]:
            """Process each dataset item."""
            # Handle table field (should be dict with 'columns' and 'data')
            table = item.get('table', '')
            table_dict = None
            if isinstance(table, dict):
                table_dict = table
                # Convert structured table to formatted string
                table = format_table(table)
            elif not isinstance(table, str):
                table = str(table)
            else:
                table = table
            
            result = {
                'table': table,  # For prompt display
                'question': str(item.get('question', '')),
                'answer': str(item.get('answer', '')),
                'qtype': item.get('qtype', ''),
                'qsubtype': item.get('qsubtype', ''),
                'id': item.get('id', ''),
                'instruction': item.get('instruction', ''),
                'instruction_type': item.get('instruction_type', instruction_type),
                'chart_type': item.get('chart_type', ''),
            }
            
            # For Visualization tasks, keep the original table dict
            if item.get('qtype') == 'Visualization' and table_dict:
                result['table_dict'] = table_dict
            
            return result


            
        
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

    @staticmethod
    def _extract_final_answer(prediction:str) -> str:
        """Extract final answer from prediction following official format.
        Expected format: "Final Answer: <answer>"
        """
        if not isinstance(prediction, str):
            prediction = str(prediction)
        
        import re
        pattern = r"Final Answer:\s*(.+?)(?:\n|$)"
        match = re.search(pattern, prediction, re.IGNORECASE)
        if match:
            return match.group(1).strip()
        
        #If no "Final Answer" found, reu
        return prediction.strip()



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
        parsed_predictions = [self._extract_final_answer(pred) for pred in predictions]

        # Extract answer strings from references if they are dicts
        ref_answers = []
        for ref in references:
            if isinstance(ref, dict):
                ref_answers.append(str(ref.get('answer', '')))
            else:
                ref_answers.append(str(ref))
        
        if self.metric == 'exact_match':
            return self._exact_match_score(parsed_predictions, ref_answers)
        elif self.metric == 'f1':
            return self._f1_score(parsed_predictions, ref_answers)
        elif self.metric == 'accuracy':
            return self._accuracy_score(parsed_predictions, ref_answers)
        else:
            return self._exact_match_score(parsed_predictions, ref_answers)

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
    @staticmethod
    def _extract_final_answer(prediction:str) -> str:
        """Extract final answer from prediction following official format.
        Expected format: "Final Answer: <answer>"
        """
        if not isinstance(prediction, str):
            prediction = str(prediction)
        
        import re
        pattern = r"Final Answer:\s*(.+?)(?:\n|$)"
        match = re.search(pattern, prediction, re.IGNORECASE)
        if match:
            return match.group(1).strip()
        return prediction.strip()

    def score(self, predictions: List, references: List, test_set: Optional[Dataset] = None) -> Dict:
        """Calculate accuracy for numeric answers."""
        if len(predictions) != len(references):
            return {
                'error': 'predictions and references have different length'
            }
        parsed_predictions = [self._extract_final_answer(pred) for pred in predictions]
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
        
        for pred, ref in zip(parsed_predictions, ref_answers):
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
            'exact_match': accuracy,
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
class TableBenchRougeEvaluator(BaseEvaluator):
    """
    ROUGE-L Evaluator for text-based answers in TableBench.
    Used for Data Analysis tasks that require descriptive answers.
    """
    
    def __init__(self):
        self.rouge = Rouge()
    
    def score(self, predictions: List, references: List, test_set: Optional[Dataset] = None) -> Dict:
        """Calculate ROUGE-L score."""
        if len(predictions) != len(references):
            return {'error': 'predictions and references have different length'}
        
        # Extract answers
        ref_answers = []
        for ref in references:
            if isinstance(ref, dict):
                ref_answers.append(str(ref.get('answer', '')))
            else:
                ref_answers.append(str(ref))
        
        # Parse predictions
        parsed_predictions = [self._extract_final_answer(pred) for pred in predictions]
        
        total_rouge_l = 0
        count = 0
        details = []
        
        for pred, ref in zip(parsed_predictions, ref_answers):
            if not pred or not ref:
                rouge_l = 0.0
            else:
                try:
                    scores = self.rouge.get_scores(pred, ref)[0]
                    rouge_l = scores['rouge-l']['f']
                except:
                    rouge_l = 0.0
            
            total_rouge_l += rouge_l
            count += 1
            details.append({'pred': pred, 'answer': ref, 'rouge_l': rouge_l})
        
        avg_rouge_l = 100 * total_rouge_l / count if count > 0 else 0
        return {
            'rouge_l': avg_rouge_l,
            'accuracy': avg_rouge_l,
            'total_count': count,
            'details': details
        }
    
    @staticmethod
    def _extract_final_answer(prediction: str) -> str:
        """Extract final answer from prediction."""
        import re
        pattern = r"Final Answer:\s*(.+?)(?:\n|$)"
        match = re.search(pattern, str(prediction), re.IGNORECASE | re.DOTALL)
        return match.group(1).strip() if match else str(prediction).strip()### 4. **关于PoT (Program of Thought) 支持**


@ICL_EVALUATORS.register_module()
class TableBenchVisualizationEvaluator(BaseEvaluator):
    """
    Evaluator for TableBench Visualization tasks.
    Returns structured results compatible with official evaluation format.
    - Each sample has 'ecr_1' (code execution success) and 'parsed_prediction' (correctness)
    - Overall metrics: ECR@1 (execution rate) and Pass (correctness rate)
    """

    def __init__(self, timeout: int = 30, use_simple_pass: bool = True) -> None:
        """
        Args:
            timeout: Maximum execution time in seconds
            use_single_pass: If True, considers successful execution as pass
            If False, Pass@1 will be 0 (chart comparison not implemented)
        """
        self.timeout = timeout
        try:
            from . import chart_metric_utils
            self.chart_utils = chart_metric_utils
        except ImportError:
            raise ImportError("chart_metric_utils module not found. Please install it using: pip install matplotlib")

    def score(self, predictions: List, references: List, test_set: Optional[Dataset] = None) -> Dict:
        """
        Calculate visualization evaluation metrics.
        Args:
            predictions: List of model predictions (code strings)
            references: List of references (dicts with 'answer' and 'chart_type')
            test_set: Optional dataset for accessing chart_type info
        Returns:
            Dict with Parse@1, ECR@1, Pass@1 metrics and detailed results
        """
        if len(predictions) != len(references):
            return {'error': 'predictions and references have different length'}

        total = len(predictions)
        parsed_count = 0
        executed_count = 0
        passed_count = 0
        details = []

        for idx, (pred, ref) in enumerate(zip(predictions, references)):
            #Initialize result structure
            parsed_result = {
                "Parse@1": False,
                "ECR@1": False,
                "Pass@1": False,
                "error": None,
            }
             # Get reference info
            # Get reference info
            if isinstance(ref, dict):
                answer = ref.get('answer', '')
                chart_type = ref.get('chart_type', '')
                table_data = ref.get('table_dict', ref.get('table', {}))  # Try table_dict first
            else:
                # Fallback: try to get from test_set
                if test_set and idx < len(test_set):
                    sample = test_set[idx]
                    answer = sample.get('answer', '')
                    chart_type = sample.get('chart_type', '')
                    table_data = sample.get('table_dict', sample.get('table', {}))
                else:
                    answer = str(ref)
                    chart_type = ''
                    table_data = {}
            
            # Step 1: Extract Python code
            python_code = self._extract_python_code(str(pred))
            
            if python_code:
                parsed_result['Parse@1'] = True
                parsed_count += 1
                
                # Step 2: Execute code and extract chart data
                success, y_predictions, error = self._execute_and_extract(
                    python_code, table_data, chart_type
                )
                
                if success:
                    parsed_result['ecr_1'] = True
                    executed_count += 1
                    
                    # Step 3: Compare with reference
                    if y_predictions is not None:
                        y_references = self._parse_answer(answer)
                        
                        if y_references is not None:
                            is_correct = self._compare_charts(
                                y_references, y_predictions, chart_type
                            )
                            
                            if is_correct:
                                parsed_result['parsed_prediction'] = True
                                passed_count += 1
                else:
                    parsed_result['error'] = str(error)[:200]
            
            detail = {
                'pred': str(pred)[:200],
                'answer': answer[:200] if answer else '',
                'chart_type': chart_type,
                'parsed_result': parsed_result
            }
            details.append(detail)
        
        # Calculate metrics
        parse_rate = 100 * parsed_count / total if total > 0 else 0
        ecr_rate = 100 * executed_count / total if total > 0 else 0
        pass_rate = 100 * passed_count / total if total > 0 else 0
        
        return {
            'Parse@1': round(parse_rate, 2),
            'ECR@1': round(ecr_rate, 2),
            'Pass@1': round(pass_rate, 2),
            'accuracy': round(pass_rate, 2),
            'total_count': total,
            'parsed_count': parsed_count,
            'executed_count': executed_count,
            'passed_count': passed_count,
            'details': details
        }


    @staticmethod
    def _extract_python_code(prediction):
        """Extract python code from prediction."""
        import re
        if not isinstance(prediction, str):
            return ""
        patterns = [
            r"\s*\n(.*?)```",
            r"```\s*\n(.*?)```",
            r"~~~python\s*\n(.*?)~~~",
            r"~~~\s*\n(.*?)~~~"
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, prediction, flags=re.DOTALL | re.IGNORECASE)
            if matches:
                return matches[-1].strip()
        
        return ""

    def _execute_and_extract(self, code: str, table_data: dict, chart_type: str) -> tuple:
        """
        Execute code and extract chart data.
        
        Returns:
            (success, y_predictions, error)
        """
        try:
            import sys
            import os
            import tempfile
            from io import StringIO
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            import pandas as pd
            
            # Create temporary directory for table.csv
            with tempfile.TemporaryDirectory() as tmpdir:
                # Save table data to CSV
                if table_data and 'columns' in table_data and 'data' in table_data:
                    df = pd.DataFrame(table_data['data'], columns=table_data['columns'])
                    table_path = os.path.join(tmpdir, 'table.csv')
                    df.to_csv(table_path, index=False)
                    
                    # Change to temp directory
                    old_cwd = os.getcwd()
                    os.chdir(tmpdir)
                else:
                    old_cwd = None
                
                try:
                    # Execute code
                    old_stdout = sys.stdout
                    sys.stdout = StringIO()
                    
                    namespace = {
                        '__builtins__': __builtins__,
                        'pd': pd,
                        'plt': plt,
                        'np': __import__('numpy')
                    }
                    
                    exec(code, namespace)
                    
                    sys.stdout = old_stdout
                    
                    # Extract chart data based on type
                    y_predictions = self._extract_chart_data(plt, chart_type)
                    
                    # Clean up
                    plt.close('all')
                    
                    if old_cwd:
                        os.chdir(old_cwd)
                    
                    return True, y_predictions, None
                    
                except Exception as e:
                    sys.stdout = old_stdout
                    if old_cwd:
                        os.chdir(old_cwd)
                    return False, None, str(e)
        
        except Exception as e:
            return False, None, str(e)
    
    def _extract_chart_data(self, plt, chart_type: str):
        """Extract y data from matplotlib object based on chart type."""
        try:
            chart_type = chart_type.lower()
            
            if chart_type == 'line':
                return self.chart_utils.get_line_y_predictions(plt)
            elif chart_type == 'bar':
                return self.chart_utils.get_bar_y_predictions(plt)
            elif chart_type == 'hbar':
                return self.chart_utils.get_hbar_y_predictions(plt)
            elif chart_type == 'pie':
                return self.chart_utils.get_pie_y_predictions(plt)
            elif chart_type == 'area':
                return self.chart_utils.get_area_y_predictions(plt)
            elif chart_type == 'radar':
                return self.chart_utils.get_radar_y_predictions(plt)
            elif chart_type == 'scatter':
                return self.chart_utils.get_scatter_y_predictions(plt)
            elif chart_type == 'waterfall':
                return self.chart_utils.get_waterfall_y_predictions(plt)
            else:
                # Try line as default
                return self.chart_utils.get_line_y_predictions(plt)
        
        except Exception as e:
            return None
    
    @staticmethod
    def _parse_answer(answer: str):
        """Parse answer string to extract y_references."""
        try:
            # answer format: "y_references = [[0.30, 1.67, ...]]"
            namespace = {}
            exec(answer, namespace)
            return namespace.get('y_references')
        except Exception:
            return None
    
    def _compare_charts(self, y_references, y_predictions, chart_type: str) -> bool:
        """Compare chart data using appropriate metric."""
        try:
            if chart_type.lower() == 'pie':
                return self.chart_utils.compute_pie_chart_metric(
                    y_references, y_predictions
                )
            else:
                return self.chart_utils.compute_general_chart_metric(
                    y_references, y_predictions
                )
        except Exception:
            return False