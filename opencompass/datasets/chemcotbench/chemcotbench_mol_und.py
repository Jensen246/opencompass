"""ChemCoTBench Molecule Understanding Dataset and Evaluator.

mol_und tasks:
- fg-level: fg_count (functional group counting), ring_count (ring unit counting)
- scaffold-level: Murcko_scaffold, ring_system_scaffold
- SMILES-level: equivalence (SMILES equivalence checking)

Reference: https://github.com/IDEA-XL/ChemCoTBench
           https://huggingface.co/datasets/OpenMol/ChemCoTBench
"""

import json
import re
from typing import Any, Dict, List, Optional

from datasets import Dataset, load_dataset

from opencompass.openicl.icl_evaluator import BaseEvaluator
from opencompass.registry import ICL_EVALUATORS, LOAD_DATASET

from ..base import BaseDataset


def _extract_answer_from_response(response: str, subtask: str) -> Optional[str]:
    """Extract answer from model response.

    Based on official evaluation logic from:
    https://github.com/IDEA-XL/ChemCoTBench/blob/main/baseline_and_eval/eval/eval_molund.py
    """
    if not response:
        return None

    response = response.strip()

    # Remove <think>...</think> part (for reasoning models like o1)
    response = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL)
    
    # Handle markdown code blocks
    if '```json' in response:
        json_match = re.search(r'```json\s*(.*?)\s*```', response, re.DOTALL)
        if json_match:
            response = json_match.group(1)
    elif '```' in response:
        json_match = re.search(r'```\s*(.*?)\s*```', response, re.DOTALL)
        if json_match:
            response = json_match.group(1)

    # Clean up response for JSON parsing
    cleaned = response.replace('\n    ', '').replace('\n', '').replace('\"', '"')

    # Try to parse as JSON
    try:
        data = json.loads(cleaned)
        if isinstance(data, dict):
            if subtask in ['fg_count', 'ring_count']:
                answer = data.get('count', data.get('answer'))
            elif subtask == 'equivalence':
                answer = data.get('output', data.get('answer'))
            elif subtask == 'Murcko_scaffold':
                answer = data.get('Output Scaffold', data.get('output', data.get('answer')))
            elif subtask == 'ring_system_scaffold':
                answer = data.get('output', data.get('answer'))
            else:
                answer = data.get('answer', data.get('output', data.get('count')))

            if answer is not None:
                return str(answer).strip()
    except json.JSONDecodeError:
        pass

    # Fallback: regex extraction based on subtask
    if subtask in ['fg_count', 'ring_count']:
        # Try to extract count field
        patterns = [
            r'"count"\s*:\s*"?(\d+)"?',
            r'"answer"\s*:\s*"?(\d+)"?',
        ]
        for pattern in patterns:
            match = re.search(pattern, response)
            if match:
                return match.group(1)
        # Last resort: find any number
        match = re.search(r'\b(\d+)\b', response)
        if match:
            return match.group(1)
            
    elif subtask in ['equivalence', 'ring_system_scaffold']:
        # Both equivalence and ring_system_scaffold use Yes/No
        patterns = [
            r'"output"\s*:\s*"(Yes|No)"',
            r'"answer"\s*:\s*"(Yes|No)"',
        ]
        for pattern in patterns:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                return match.group(1)
        # Last resort
        if re.search(r'\byes\b', response, re.IGNORECASE):
            return 'Yes'
        if re.search(r'\bno\b', response, re.IGNORECASE):
            return 'No'
            
    elif subtask == 'Murcko_scaffold':
        # For Murcko scaffold tasks, extract SMILES
        patterns = [
            r'"Output Scaffold"\s*:\s*"([^"]*)"',
            r'"output"\s*:\s*"([^"]*)"',
            r'"answer"\s*:\s*"([^"]*)"',
        ]
        for pattern in patterns:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                return match.group(1).strip()

    return response.strip()


def _calculate_tanimoto_similarity(smiles1: str, smiles2: str) -> float:
    """Calculate Tanimoto similarity between two SMILES using Morgan fingerprints.
    
    Args:
        smiles1: First SMILES string (e.g., predicted scaffold)
        smiles2: Second SMILES string (e.g., ground truth scaffold)
    
    Returns:
        Tanimoto similarity score (0.0 to 1.0)
    """
    try:
        from rdkit import Chem, DataStructs
        from rdkit.Chem import AllChem
        
        mol1 = Chem.MolFromSmiles(smiles1)
        mol2 = Chem.MolFromSmiles(smiles2)
        
        if mol1 is None or mol2 is None:
            return 0.0
        
        # Calculate Morgan fingerprint (radius=2, 1024 bits)
        fp1 = AllChem.GetMorganFingerprintAsBitVect(mol1, 2, nBits=1024)
        fp2 = AllChem.GetMorganFingerprintAsBitVect(mol2, 2, nBits=1024)
        
        return DataStructs.TanimotoSimilarity(fp1, fp2)
    except ImportError:
        # Fallback: exact string match
        return 1.0 if smiles1 == smiles2 else 0.0
    except Exception:
        return 0.0


@LOAD_DATASET.register_module()
class ChemCoTBenchMolUndDataset(BaseDataset):
    """ChemCoTBench Molecule Understanding Dataset.

    This dataset loads the mol_und tasks from ChemCoTBench benchmark.
    Subtasks: fg_count, ring_count, Murcko_scaffold, ring_system_scaffold, equivalence
    """

    @staticmethod
    def load(path: str = 'OpenMol/ChemCoTBench',
             subtask: str = 'fg_count',
             seed: int = 42,
             **kwargs) -> Dataset:
        """Load and preprocess the mol_und dataset from HuggingFace."""
        ds = load_dataset(path, split='train', trust_remote_code=True)
        ds = ds.filter(lambda x: x['task'] == 'mol_und' and x['subtask'] == subtask)

        def process_item(item: Dict[str, Any]) -> Dict[str, Any]:
            # For scaffold tasks, we may need source SMILES for evaluation
            meta = item.get('meta', '{}')
            if isinstance(meta, str):
                try:
                    meta = json.loads(meta)
                except json.JSONDecodeError:
                    meta = {}
            
            source_smiles = meta.get('source_smiles', meta.get('source', ''))
            
            return {
                'prompt': item['query'],
                'answer': str(item['gt']),
                'source_smiles': source_smiles,
                'identifier': item.get('id', ''),
                'subtask': subtask,
            }

        processed_ds = ds.map(process_item, remove_columns=ds.column_names)
        processed_ds = processed_ds.shuffle(seed=seed)
        return processed_ds


@ICL_EVALUATORS.register_module()
class ChemCoTBenchMolUndEvaluator(BaseEvaluator):
    """Evaluator for ChemCoTBench mol_und task.

    Evaluation metrics based on official paper:
    - fg_count, ring_count: Mean Absolute Error (MAE)
    - equivalence, ring_system_scaffold: Accuracy (Yes/No match with gt)
    - Murcko_scaffold: Tanimoto similarity (Morgan fingerprint)
    """

    def __init__(self, subtask: str = 'fg_count') -> None:
        super().__init__()
        self.subtask = subtask

    def score(self, predictions: List[str], references: List[str],
              test_set: Optional[Dataset] = None) -> Dict[str, Any]:
        """Compute evaluation metrics."""
        if len(predictions) != len(references):
            return {'error': 'predictions and references have different length'}

        total = len(predictions)
        
        if self.subtask in ['fg_count', 'ring_count']:
            # MAE evaluation (lower is better)
            mae_sum = 0
            for pred_raw, ref in zip(predictions, references):
                pred_extracted = _extract_answer_from_response(pred_raw, self.subtask)
                try:
                    pred_val = int(pred_extracted) if pred_extracted else 0
                    ref_val = int(ref)
                    mae_sum += abs(pred_val - ref_val)
                except (ValueError, TypeError):
                    mae_sum += abs(int(ref))  # Treat invalid prediction as 0
            
            return {'mae': mae_sum / total if total > 0 else 0}
            
        elif self.subtask in ['equivalence', 'ring_system_scaffold']:
            # Yes/No accuracy
            correct = 0
            for pred_raw, ref in zip(predictions, references):
                pred_extracted = _extract_answer_from_response(pred_raw, self.subtask)
                if str(pred_extracted).lower() == str(ref).lower():
                    correct += 1
            
            return {'accuracy': correct / total * 100 if total > 0 else 0}
            
        elif self.subtask == 'Murcko_scaffold':
            # Tanimoto similarity between predicted scaffold and ground truth scaffold
            similarity_sum = 0.0
            
            for pred_raw, ref in zip(predictions, references):
                pred_extracted = _extract_answer_from_response(pred_raw, self.subtask)
                
                if pred_extracted and ref:
                    similarity = _calculate_tanimoto_similarity(pred_extracted, ref)
                    similarity_sum += similarity
            
            return {'tanimoto_similarity_larger_means_better': similarity_sum / total if total > 0 else 0}
        
        # Default: exact match accuracy
        correct = 0
        for pred_raw, ref in zip(predictions, references):
            pred_extracted = _extract_answer_from_response(pred_raw, self.subtask)
            if str(pred_extracted).lower() == str(ref).lower():
                correct += 1
        
        return {'accuracy': correct / total * 100 if total > 0 else 0}
