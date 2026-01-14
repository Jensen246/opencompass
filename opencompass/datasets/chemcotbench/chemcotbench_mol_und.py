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
            elif subtask in ['Murcko_scaffold', 'ring_system_scaffold']:
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
            r'"output"\s*:\s*"([^"]*)"',
            r'"answer"\s*:\s*"([^"]*)"',
        ]
        for pattern in patterns:
            match = re.search(pattern, response)
            if match:
                return match.group(1).strip()

    return response.strip()


def _calculate_scaffold_similarity(smiles1: str, smiles2: str) -> tuple:
    """Calculate scaffold consistency (hard and soft).
    
    Based on official evaluation using rdFMCS.FindMCS:
    https://github.com/IDEA-XL/ChemCoTBench/blob/main/baseline_and_eval/eval/eval_metric.py
    
    Returns (is_same_scaffold, similarity_score)
    """
    try:
        from rdkit import Chem, DataStructs
        from rdkit.Chem import AllChem, rdFMCS
        from rdkit.Chem.Scaffolds.MurckoScaffold import MurckoScaffoldSmiles
        
        # Get Murcko scaffolds
        try:
            scaffold1 = MurckoScaffoldSmiles(smiles1)
            scaffold2 = MurckoScaffoldSmiles(smiles2)
        except Exception:
            return False, 0.0
        
        # Exact match (hard)
        if scaffold1 == scaffold2:
            return True, 1.0
        
        # Soft: use MCS + fingerprint similarity (official implementation)
        mol1 = Chem.MolFromSmiles(scaffold1)
        mol2 = Chem.MolFromSmiles(scaffold2)
        
        if mol1 is None or mol2 is None:
            return False, 0.0
        
        # Find MCS first (as in official implementation)
        mcs = rdFMCS.FindMCS([mol1, mol2])
        mcs_mol = Chem.MolFromSmarts(mcs.smartsString) if mcs.numAtoms > 0 else None
        
        if mcs_mol:
            # Calculate Morgan fingerprint similarity
            fp1 = AllChem.GetMorganFingerprintAsBitVect(mol1, 2, nBits=1024)
            fp2 = AllChem.GetMorganFingerprintAsBitVect(mol2, 2, nBits=1024)
            similarity = DataStructs.TanimotoSimilarity(fp1, fp2)
        else:
            similarity = 0.0
        
        return False, similarity
    except ImportError:
        return smiles1 == smiles2, 1.0 if smiles1 == smiles2 else 0.0
    except Exception:
        return False, 0.0


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

    Evaluation metrics based on official implementation:
    https://github.com/IDEA-XL/ChemCoTBench/blob/main/baseline_and_eval/eval/eval_molund.py
    
    - fg_count, ring_count: Mean Absolute Error (MAE) - score = sum(|pred-gt|) / len
    - equivalence: Accuracy (Yes/No match with gt)
    - ring_system_scaffold: Rate of "Yes" predictions
    - Murcko_scaffold: Scaffold consistency (soft score using MCS + Morgan fingerprint)
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
        details = []
        
        if self.subtask in ['fg_count', 'ring_count']:
            # MAE evaluation: score = sum(|pred-gt|) / len (lower is better)
            # Based on official: score = sum([abs(int(pred_list[i])-int(gt_list[i])) for i in range(len(pred_list))]) / len(gt_list)
            mae_sum = 0
            valid_count = 0
            
            for pred_raw, ref in zip(predictions, references):
                pred_extracted = _extract_answer_from_response(pred_raw, self.subtask)
                
                try:
                    pred_val = int(pred_extracted) if pred_extracted else None
                    ref_val = int(ref)
                    
                    if pred_val is not None:
                        error = abs(pred_val - ref_val)
                        mae_sum += error
                        valid_count += 1
                        is_correct = (error == 0)
                    else:
                        error = None
                        is_correct = False
                except (ValueError, TypeError):
                    error = None
                    is_correct = False
                    pred_val = None
                    ref_val = ref
                
                details.append({
                    'pred_raw': pred_raw[:500] if pred_raw else '',
                    'pred_extracted': pred_extracted,
                    'pred_value': pred_val,
                    'gold': ref,
                    'error': error,
                    'correct': is_correct,
                })
            
            # Official: score = sum(abs) / len (using total, not valid_count)
            mae = mae_sum / total if total > 0 else 0
            accuracy = sum(1 for d in details if d['correct']) / total * 100 if total > 0 else 0
            valid_rate = valid_count / total * 100 if total > 0 else 0
            
            return {
                'mae': mae,
                'accuracy': accuracy,
                'valid_rate': valid_rate,
                'total_count': total,
                'details': details,
            }
            
        elif self.subtask == 'equivalence':
            # Yes/No accuracy: compare pred.lower() == gt.lower()
            # Based on official: count = sum(1 for i if str(pred_list[i]).lower() == str(gt_list[i]).lower())
            correct = 0
            
            for pred_raw, ref in zip(predictions, references):
                pred_extracted = _extract_answer_from_response(pred_raw, self.subtask)
                
                pred_norm = str(pred_extracted).lower() if pred_extracted else ''
                ref_norm = str(ref).lower()
                
                is_correct = pred_norm == ref_norm
                if is_correct:
                    correct += 1
                
                details.append({
                    'pred_raw': pred_raw[:500] if pred_raw else '',
                    'pred_extracted': pred_extracted,
                    'pred_normalized': pred_norm,
                    'gold': ref,
                    'gold_normalized': ref_norm,
                    'correct': is_correct,
                })
            
            accuracy = correct / total * 100 if total > 0 else 0
            
            return {
                'accuracy': accuracy,
                'correct_count': correct,
                'total_count': total,
                'details': details,
            }
        
        elif self.subtask == 'ring_system_scaffold':
            # Based on official: count = sum(1 for item in pred_list if str(item).lower() == "yes")
            # score = count / len(pred_list)
            yes_count = 0
            valid_count = 0
            
            for pred_raw, ref in zip(predictions, references):
                pred_extracted = _extract_answer_from_response(pred_raw, self.subtask)
                
                if pred_extracted:
                    valid_count += 1
                    is_yes = str(pred_extracted).lower() == 'yes'
                    if is_yes:
                        yes_count += 1
                else:
                    is_yes = False
                
                details.append({
                    'pred_raw': pred_raw[:500] if pred_raw else '',
                    'pred_extracted': pred_extracted,
                    'gold': ref,
                    'is_yes': is_yes,
                })
            
            # Score is the rate of "Yes" predictions
            score = yes_count / valid_count * 100 if valid_count > 0 else 0
            valid_rate = valid_count / total * 100 if total > 0 else 0
            
            return {
                'score': score,
                'yes_count': yes_count,
                'valid_rate': valid_rate,
                'total_count': total,
                'details': details,
            }
            
        elif self.subtask == 'Murcko_scaffold':
            # Scaffold consistency (soft score)
            # Based on official: scaffold_hard, scaffold_soft = prop_evaluater.scaffold_consistency(...)
            # score = scaffold_soft / len(pred_list)
            scaffold_hard_count = 0
            scaffold_soft_sum = 0.0
            valid_count = 0
            
            # Get source SMILES from test_set if available
            source_list = []
            if test_set is not None:
                for item in test_set:
                    source_list.append(item.get('source_smiles', ''))
            else:
                source_list = references  # Fall back to using references as gt molecules
            
            for i, (pred_raw, ref) in enumerate(zip(predictions, references)):
                pred_extracted = _extract_answer_from_response(pred_raw, self.subtask)
                source = source_list[i] if i < len(source_list) else ref
                
                if pred_extracted and source:
                    is_same, similarity = _calculate_scaffold_similarity(source, pred_extracted)
                    scaffold_soft_sum += similarity
                    if is_same:
                        scaffold_hard_count += 1
                    valid_count += 1
                else:
                    is_same = False
                    similarity = 0.0
                
                details.append({
                    'pred_raw': pred_raw[:500] if pred_raw else '',
                    'pred_extracted': pred_extracted,
                    'source': source,
                    'gold': ref,
                    'scaffold_same': is_same,
                    'scaffold_similarity': similarity,
                })
            
            # Official: score = scaffold_soft / len(pred_list)
            avg_score = scaffold_soft_sum / total * 100 if total > 0 else 0
            scaffold_hard_rate = scaffold_hard_count / total * 100 if total > 0 else 0
            valid_rate = valid_count / total * 100 if total > 0 else 0
            
            return {
                'scaffold_score': avg_score,
                'scaffold_hard': scaffold_hard_rate,
                'valid_rate': valid_rate,
                'total_count': total,
                'details': details,
            }
        
        # Default: exact match
        correct = 0
        for pred_raw, ref in zip(predictions, references):
            pred_extracted = _extract_answer_from_response(pred_raw, self.subtask)
            if str(pred_extracted).lower() == str(ref).lower():
                correct += 1
            details.append({
                'pred_raw': pred_raw[:500] if pred_raw else '',
                'pred_extracted': pred_extracted,
                'gold': ref,
                'correct': str(pred_extracted).lower() == str(ref).lower(),
            })
        
        return {
            'accuracy': correct / total * 100 if total > 0 else 0,
            'correct_count': correct,
            'total_count': total,
            'details': details,
        }
