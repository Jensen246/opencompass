"""ChemCoTBench Molecule Editing Dataset and Evaluator.

mol_edit tasks:
- add: Add functional groups to molecules
- delete: Delete functional groups from molecules
- sub: Substitute functional groups in molecules

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


# Functional group SMARTS patterns from official implementation
GROUP_TO_SMARTS = {
    'benzene': '[cR1]1[cR1][cR1][cR1][cR1][cR1]1',
    'benzene_ring': '[cR1]1[cR1][cR1][cR1][cR1][cR1]1',
    'hydroxyl': '[OX2H]',
    'aldehyde': '[CX3H1](=O)[#6]',
    'ketone': '[#6][CX3](=O)[#6]',
    'carboxyl': '[CX3](=O)[OX2H1]',
    'ester': '[#6][CX3](=O)[OX2H0][#6]',
    'anhydride': '[CX3](=[OX1])[OX2][CX3](=[OX1])',
    'amine': '[NX3;H2,H1;!$(NC=O)]',
    'amide': '[NX3][CX3](=[OX1])[#6]',
    'nitro': '[$([NX3](=O)=O),$([NX3+](=O)[O-])][!#8]',
    'halo': '[F,Cl,Br,I]',
    'thiol': '[#16X2H]',
    'thioether': '[SX2][CX4]',
    'disulfide': '[#16X2H0][#16X2H0]',
    'sulfoxide': '[$([#16X3]=[OX1]),$([#16X3+][OX1-])]',
    'sulfone': '[$([#16X4](=[OX1])=[OX1]),$([#16X4+2]([OX1-])[OX1-])]',
    'sulfide': '[#16X2H0]',
    'nitrile': '[NX1]#[CX2]',
    'borane': '[BX3]',
}

GROUP_SET = set(GROUP_TO_SMARTS.keys())


def _extract_smiles_from_response(response: str) -> Optional[str]:
    """Extract SMILES from model response."""
    if not response:
        return None

    response = response.strip()

    # Remove <think>...</think> part
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

    # Clean up for JSON parsing
    cleaned = response.replace('\n    ', '').replace('\n', '').replace('\"', '"')

    try:
        data = json.loads(cleaned)
        if isinstance(data, dict):
            answer = data.get('output', data.get('answer'))
            if answer is not None:
                return str(answer).strip()
    except json.JSONDecodeError:
        pass

    # Fallback: regex extraction
    patterns = [
        r'"output"\s*:\s*"([^"]*)"',
        r'"answer"\s*:\s*"([^"]*)"',
    ]

    for pattern in patterns:
        match = re.search(pattern, response, re.IGNORECASE)
        if match:
            return match.group(1).strip()

    # Last resort: find SMILES-like string
    smiles_pattern = r'[A-Za-z][A-Za-z0-9@\[\]\(\)=#\-\+\.\\\/]{4,}'
    matches = re.findall(smiles_pattern, response)
    if matches:
        return max(matches, key=len)

    return response.strip()


def _is_valid_smiles(smiles: str) -> bool:
    """Check if SMILES is valid."""
    if not smiles:
        return False
    try:
        from rdkit import Chem
        mol = Chem.MolFromSmiles(smiles)
        return mol is not None
    except ImportError:
        return True
    except Exception:
        return False


def _count_functional_group(smiles: str, group: str) -> Optional[int]:
    """Count occurrences of a functional group in a molecule.
    
    Based on official implementation:
    https://github.com/IDEA-XL/ChemCoTBench/blob/main/baseline_and_eval/eval/eval_moledit.py
    """
    try:
        from rdkit import Chem
        
        if group not in GROUP_TO_SMARTS:
            return None
            
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
            
        smarts = GROUP_TO_SMARTS[group]
        pattern = Chem.MolFromSmarts(smarts)
        if pattern is None:
            return None
            
        matches = mol.GetSubstructMatches(pattern)
        count = len(matches)
        
        # Special handling for sulfide (exclude disulfides)
        if group == 'sulfide':
            disulfide_pattern = Chem.MolFromSmarts('[#16X2H0][#16X2H0]')
            disulfide_matches = mol.GetSubstructMatches(disulfide_pattern)
            count -= len(disulfide_matches)
        
        return count
    except ImportError:
        return None
    except Exception:
        return None


def _check_edit_add_valid(src: str, tgt: str, group: str) -> bool:
    """Check if add operation is valid."""
    if group not in GROUP_SET:
        return False
    if not _is_valid_smiles(src) or not _is_valid_smiles(tgt):
        return False
    
    src_count = _count_functional_group(src, group)
    tgt_count = _count_functional_group(tgt, group)
    
    if src_count is None or tgt_count is None:
        return False
    
    return tgt_count == src_count + 1


def _check_edit_del_valid(src: str, tgt: str, group: str) -> bool:
    """Check if delete operation is valid."""
    if group not in GROUP_SET:
        return False
    if not _is_valid_smiles(src) or not _is_valid_smiles(tgt):
        return False
    
    src_count = _count_functional_group(src, group)
    tgt_count = _count_functional_group(tgt, group)
    
    if src_count is None or tgt_count is None:
        return False
    
    return tgt_count == src_count - 1


def _check_edit_sub_valid(src: str, tgt: str, remove_group: str, add_group: str) -> bool:
    """Check if substitution operation is valid."""
    if remove_group not in GROUP_SET or add_group not in GROUP_SET:
        return False
    if not _is_valid_smiles(src) or not _is_valid_smiles(tgt):
        return False
    
    src_remove_count = _count_functional_group(src, remove_group)
    tgt_remove_count = _count_functional_group(tgt, remove_group)
    src_add_count = _count_functional_group(src, add_group)
    tgt_add_count = _count_functional_group(tgt, add_group)
    
    if any(c is None for c in [src_remove_count, tgt_remove_count, src_add_count, tgt_add_count]):
        return False
    
    return (tgt_remove_count == src_remove_count - 1 and 
            tgt_add_count == src_add_count + 1)


def _calculate_similarity(smiles1: str, smiles2: str) -> float:
    """Calculate Tanimoto similarity using Morgan fingerprints."""
    try:
        from rdkit import Chem, DataStructs
        from rdkit.Chem import AllChem
        
        mol1 = Chem.MolFromSmiles(smiles1)
        mol2 = Chem.MolFromSmiles(smiles2)
        
        if mol1 is None or mol2 is None:
            return 0.0
        
        fp1 = AllChem.GetMorganFingerprintAsBitVect(mol1, radius=2, nBits=2048)
        fp2 = AllChem.GetMorganFingerprintAsBitVect(mol2, radius=2, nBits=2048)
        
        return DataStructs.TanimotoSimilarity(fp1, fp2)
    except ImportError:
        return 1.0 if smiles1 == smiles2 else 0.0
    except Exception:
        return 0.0


def _extract_source_smiles_from_query(query: str) -> Optional[str]:
    """Extract source molecule SMILES from the query text."""
    # Pattern: "Input Molecule: <SMILES>," or "Molecule SMILES: <SMILES>"
    patterns = [
        r'Input Molecule:\s*([^\s,]+)',
        r'Molecule SMILES:\s*([^\s,]+)',
        r'Source Molecule:\s*([^\s,]+)',
    ]
    for pattern in patterns:
        match = re.search(pattern, query)
        if match:
            return match.group(1).strip()
    return None


@LOAD_DATASET.register_module()
class ChemCoTBenchMolEditDataset(BaseDataset):
    """ChemCoTBench Molecule Editing Dataset."""

    @staticmethod
    def load(path: str = 'OpenMol/ChemCoTBench',
             subtask: str = 'add',
             **kwargs) -> Dataset:
        """Load and preprocess the mol_edit dataset from HuggingFace."""
        ds = load_dataset(path, split='train', trust_remote_code=True)
        ds = ds.filter(lambda x: x['task'] == 'mol_edit' and x['subtask'] == subtask)

        def process_item(item: Dict[str, Any]) -> Dict[str, Any]:
            meta = item.get('meta', '{}')
            if isinstance(meta, str):
                try:
                    meta = json.loads(meta)
                except json.JSONDecodeError:
                    meta = {}

            # Extract source SMILES from query
            source_smiles = _extract_source_smiles_from_query(item['query'])
            reference = meta.get('reference', '')
            
            # For add task: added_group
            # For delete task: removed_group
            # For sub task: added_group (target) and removed_group (source)
            if subtask == 'add':
                group_a = meta.get('added_group', '')
                group_b = ''
            elif subtask == 'delete':
                group_a = meta.get('removed_group', '')
                group_b = ''
            elif subtask == 'sub':
                group_a = meta.get('added_group', '')  # Group to add
                group_b = meta.get('removed_group', '')  # Group to remove

            return {
                'prompt': item['query'],
                'answer': reference,
                'source_smiles': source_smiles or '',
                'group_a': group_a,
                'group_b': group_b,
                'identifier': item.get('id', ''),
                'subtask': subtask,
            }

        processed_ds = ds.map(process_item, remove_columns=ds.column_names)
        return processed_ds


@ICL_EVALUATORS.register_module()
class ChemCoTBenchMolEditEvaluator(BaseEvaluator):
    """Evaluator for ChemCoTBench mol_edit task.

    Evaluation based on official implementation:
    https://github.com/IDEA-XL/ChemCoTBench/blob/main/baseline_and_eval/eval/eval_moledit.py
    
    - correct_rate: Whether the edit operation is correctly performed
    - valid_rate: Rate of valid SMILES extraction
    """

    def __init__(self, subtask: str = 'add') -> None:
        super().__init__()
        self.subtask = subtask

    def score(self, predictions: List[str], references: List[str],
              test_set: Optional[Dataset] = None) -> Dict[str, Any]:
        """Compute evaluation metrics."""
        if len(predictions) != len(references):
            return {'error': 'predictions and references have different length'}

        total = len(predictions)
        correct_count = 0
        valid_count = 0
        total_similarity = 0.0
        details = []

        # Get additional info from test_set
        source_list = []
        group_a_list = []
        group_b_list = []
        
        if test_set is not None:
            for item in test_set:
                source_list.append(item.get('source_smiles', ''))
                group_a_list.append(item.get('group_a', ''))
                group_b_list.append(item.get('group_b', ''))

        for i, (pred_raw, ref) in enumerate(zip(predictions, references)):
            pred_smiles = _extract_smiles_from_response(pred_raw)
            
            source = source_list[i] if i < len(source_list) else ''
            group_a = group_a_list[i] if i < len(group_a_list) else ''
            group_b = group_b_list[i] if i < len(group_b_list) else ''
            
            # Check validity
            pred_valid = _is_valid_smiles(pred_smiles) if pred_smiles else False
            if pred_valid:
                valid_count += 1
            
            # Check correctness based on subtask
            is_correct = False
            if pred_smiles and source and group_a:
                if self.subtask == 'add':
                    is_correct = _check_edit_add_valid(source, pred_smiles, group_a)
                elif self.subtask == 'delete':
                    is_correct = _check_edit_del_valid(source, pred_smiles, group_a)
                elif self.subtask == 'sub' and group_b:
                    is_correct = _check_edit_sub_valid(source, pred_smiles, group_b, group_a)
            
            if is_correct:
                correct_count += 1
            
            # Calculate similarity with reference
            similarity = 0.0
            if pred_smiles and ref:
                similarity = _calculate_similarity(pred_smiles, ref)
            total_similarity += similarity

            details.append({
                'pred_raw': pred_raw[:500] if pred_raw else '',
                'pred_smiles': pred_smiles,
                'source': source,
                'gold': ref,
                'group_a': group_a,
                'group_b': group_b,
                'pred_valid': pred_valid,
                'correct': is_correct,
                'tanimoto_similarity': similarity,
            })

        correct_rate = correct_count / total * 100 if total > 0 else 0
        valid_rate = valid_count / total * 100 if total > 0 else 0
        avg_similarity = total_similarity / total * 100 if total > 0 else 0

        return {
            'correct_rate': correct_rate,
            'valid_rate': valid_rate,
            'tanimoto_similarity': avg_similarity,
            'correct_count': correct_count,
            'valid_count': valid_count,
            'total_count': total,
            'details': details,
        }
