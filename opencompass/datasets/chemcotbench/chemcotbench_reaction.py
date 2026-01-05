"""ChemCoTBench Reaction Dataset and Evaluator.

reaction tasks:
- fs: Forward synthesis (forward reaction prediction)
- retro: Retrosynthesis prediction (single-step)
- rcr: Reaction condition recommendation
- nepp: Next elementary step product prediction
- mechsel: Mechanism route selection

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
    """Extract answer from model response based on subtask type."""
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
            if subtask == 'mechsel':
                choice = data.get('choice', data.get('answer'))
                if choice:
                    return str(choice).strip().upper()
            elif subtask == 'fs':
                return data.get('Major Product', data.get('output', data.get('answer')))
            elif subtask == 'retro':
                return data.get('Reactants', data.get('output', data.get('answer')))
            elif subtask == 'rcr':
                return data.get('condition', data.get('answer'))
            elif subtask == 'nepp':
                return data.get('output', data.get('answer'))
            else:
                return data.get('answer', data.get('output'))
    except json.JSONDecodeError:
        pass

    # Fallback: regex extraction based on subtask
    if subtask == 'mechsel':
        patterns = [
            r'"choice"\s*:\s*"([A-Z])"',
            r'"answer"\s*:\s*"([A-Z])"',
        ]
        for pattern in patterns:
            match = re.search(pattern, response)
            if match:
                return match.group(1).upper()
        # Last resort: find any single letter
        match = re.search(r'\b([A-Z])\b', response)
        if match:
            return match.group(1)
            
    elif subtask == 'fs':
        patterns = [
            r'"Major Product"\s*:\s*"([^"]*)"',
            r'"output"\s*:\s*"([^"]*)"',
            r'"answer"\s*:\s*"([^"]*)"',
        ]
        for pattern in patterns:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                return match.group(1).strip()
                
    elif subtask == 'retro':
        patterns = [
            r'"Reactants"\s*:\s*"([^"]*)"',
            r'"output"\s*:\s*"([^"]*)"',
            r'"answer"\s*:\s*"([^"]*)"',
        ]
        for pattern in patterns:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                return match.group(1).strip()

    return response.strip()


def _convert_to_canonical_smiles(smiles: str) -> Optional[str]:
    """Convert SMILES to canonical form.
    
    Based on official implementation:
    https://github.com/IDEA-XL/ChemCoTBench/blob/main/baseline_and_eval/evaluator.py
    """
    if not smiles:
        return None
    try:
        from rdkit import Chem
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            # Use isomericSmiles=False to match official implementation
            return Chem.MolToSmiles(mol, isomericSmiles=False, canonical=True)
        return None
    except ImportError:
        return smiles
    except Exception:
        return None


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


def _exact_match_smiles(smiles1: str, smiles2: str) -> bool:
    """Check if two SMILES represent the same molecule using InChI.
    
    Based on official implementation:
    https://github.com/IDEA-XL/ChemCoTBench/blob/main/baseline_and_eval/evaluator.py
    """
    try:
        from rdkit import Chem
        
        mol1 = Chem.MolFromSmiles(smiles1)
        mol2 = Chem.MolFromSmiles(smiles2)
        
        if mol1 is None or mol2 is None:
            return False
        
        inchi1 = Chem.MolToInchi(mol1)
        inchi2 = Chem.MolToInchi(mol2)
        
        return inchi1 == inchi2
    except Exception:
        return False


def _calculate_morgan_similarity(smiles1: str, smiles2: str, radius: int = 2) -> float:
    """Calculate Morgan fingerprint similarity.
    
    Uses GetMorganFingerprint (not BitVect) to match official implementation.
    Reference: https://github.com/IDEA-XL/ChemCoTBench/blob/main/baseline_and_eval/evaluator.py
    """
    try:
        from rdkit import Chem, DataStructs
        from rdkit.Chem import AllChem
        
        mol1 = Chem.MolFromSmiles(smiles1)
        mol2 = Chem.MolFromSmiles(smiles2)
        
        if mol1 is None or mol2 is None:
            return 0.0
        
        # Use GetMorganFingerprint (count-based) as in official implementation
        fp1 = AllChem.GetMorganFingerprint(mol1, radius)
        fp2 = AllChem.GetMorganFingerprint(mol2, radius)
        
        return DataStructs.TanimotoSimilarity(fp1, fp2)
    except Exception:
        return 0.0


def _calculate_maccs_similarity(smiles1: str, smiles2: str) -> float:
    """Calculate MACCS keys similarity.
    
    Reference: https://github.com/IDEA-XL/ChemCoTBench/blob/main/baseline_and_eval/evaluator.py
    """
    try:
        from rdkit import Chem, DataStructs
        from rdkit.Chem import MACCSkeys
        
        mol1 = Chem.MolFromSmiles(smiles1)
        mol2 = Chem.MolFromSmiles(smiles2)
        
        if mol1 is None or mol2 is None:
            return 0.0
        
        fp1 = MACCSkeys.GenMACCSKeys(mol1)
        fp2 = MACCSkeys.GenMACCSKeys(mol2)
        
        return DataStructs.FingerprintSimilarity(fp1, fp2, metric=DataStructs.TanimotoSimilarity)
    except Exception:
        return 0.0


def _calculate_rdk_similarity(smiles1: str, smiles2: str) -> float:
    """Calculate RDKit fingerprint similarity.
    
    Reference: https://github.com/IDEA-XL/ChemCoTBench/blob/main/baseline_and_eval/evaluator.py
    """
    try:
        from rdkit import Chem, DataStructs
        
        mol1 = Chem.MolFromSmiles(smiles1)
        mol2 = Chem.MolFromSmiles(smiles2)
        
        if mol1 is None or mol2 is None:
            return 0.0
        
        fp1 = Chem.RDKFingerprint(mol1)
        fp2 = Chem.RDKFingerprint(mol2)
        
        return DataStructs.FingerprintSimilarity(fp1, fp2, metric=DataStructs.TanimotoSimilarity)
    except Exception:
        return 0.0


def _calculate_levenshtein(s1: str, s2: str) -> int:
    """Calculate Levenshtein distance between two strings.
    
    Reference: https://github.com/IDEA-XL/ChemCoTBench/blob/main/baseline_and_eval/evaluator.py
    """
    try:
        from Levenshtein import distance
        return distance(s1, s2)
    except ImportError:
        # Fallback implementation
        if len(s1) < len(s2):
            s1, s2 = s2, s1
        if len(s2) == 0:
            return len(s1)
        
        prev_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            curr_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = prev_row[j + 1] + 1
                deletions = curr_row[j] + 1
                substitutions = prev_row[j] + (c1 != c2)
                curr_row.append(min(insertions, deletions, substitutions))
            prev_row = curr_row
        return prev_row[-1]


def _calculate_bleu(pred: str, gt: str) -> float:
    """Calculate character-level BLEU score.
    
    Reference: https://github.com/IDEA-XL/ChemCoTBench/blob/main/baseline_and_eval/evaluator.py
    """
    if not pred or not gt:
        return 0.0
    
    try:
        from nltk.translate.bleu_score import corpus_bleu
        
        # Character-level tokenization as in official implementation
        gt_tokens = [[c for c in gt]]
        pred_tokens = [c for c in pred]
        
        return corpus_bleu([gt_tokens], [pred_tokens])
    except ImportError:
        return 0.0
    except Exception:
        return 0.0


@LOAD_DATASET.register_module()
class ChemCoTBenchReactionDataset(BaseDataset):
    """ChemCoTBench Reaction Dataset."""

    @staticmethod
    def load(path: str = 'OpenMol/ChemCoTBench',
             subtask: str = 'fs',
             **kwargs) -> Dataset:
        """Load and preprocess the reaction dataset from HuggingFace."""
        ds = load_dataset(path, split='train', trust_remote_code=True)
        ds = ds.filter(lambda x: x['task'] == 'reaction' and x['subtask'] == subtask)

        def process_item(item: Dict[str, Any]) -> Dict[str, Any]:
            gt = item.get('gt', '')

            # For fs task, gt may be a JSON string
            answer = gt
            if subtask == 'fs' and isinstance(gt, str) and gt.startswith('{'):
                try:
                    gt_data = json.loads(gt)
                    answer = gt_data.get('Major Product', '')
                except json.JSONDecodeError:
                    answer = gt

            return {
                'prompt': item['query'],
                'answer': str(answer) if answer else '',
                'gt_full': gt,
                'identifier': item.get('id', ''),
                'subtask': subtask,
            }

        processed_ds = ds.map(process_item, remove_columns=ds.column_names)
        return processed_ds


@ICL_EVALUATORS.register_module()
class ChemCoTBenchReactionEvaluator(BaseEvaluator):
    """Evaluator for ChemCoTBench reaction task.

    Based on official implementation from:
    https://github.com/IDEA-XL/ChemCoTBench/blob/main/baseline_and_eval/evaluator.py
    
    Different metrics for different subtasks:
    - fs/retro/nepp: exact_match, validity, morgan_sims, maccs_sims, rdk_sims, bleu, levenshtein
    - rcr: exact match on condition text
    - mechsel: accuracy on choice letter
    """

    def __init__(self, subtask: str = 'fs') -> None:
        super().__init__()
        self.subtask = subtask

    def score(self, predictions: List[str], references: List[str],
              test_set: Optional[Dataset] = None) -> Dict[str, Any]:
        """Compute evaluation metrics."""
        if len(predictions) != len(references):
            return {'error': 'predictions and references have different length'}

        if self.subtask == 'mechsel':
            return self._score_classification(predictions, references)
        elif self.subtask == 'rcr':
            return self._score_condition(predictions, references)
        else:
            return self._score_smiles(predictions, references)

    def _score_classification(self, predictions: List[str],
                             references: List[str]) -> Dict[str, Any]:
        """Score for classification tasks (mechsel)."""
        total = len(predictions)
        correct = 0
        details = []

        for pred_raw, ref in zip(predictions, references):
            pred = _extract_answer_from_response(pred_raw, self.subtask)

            # Normalize: extract first letter
            pred_norm = ''
            if pred:
                pred_str = str(pred).upper().strip()
                match = re.search(r'[A-Z]', pred_str)
                if match:
                    pred_norm = match.group()

            ref_norm = str(ref).upper().strip()
            match = re.search(r'[A-Z]', ref_norm)
            if match:
                ref_norm = match.group()

            is_correct = pred_norm == ref_norm
            if is_correct:
                correct += 1

            details.append({
                'pred_raw': pred_raw[:500] if pred_raw else '',
                'pred_extracted': pred,
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

    def _score_condition(self, predictions: List[str],
                        references: List[str]) -> Dict[str, Any]:
        """Score for reaction condition tasks (rcr)."""
        total = len(predictions)
        exact_matches = 0
        details = []

        for pred_raw, ref in zip(predictions, references):
            pred = _extract_answer_from_response(pred_raw, self.subtask)

            pred_norm = str(pred).lower().strip() if pred else ''
            ref_norm = str(ref).lower().strip()

            is_exact = pred_norm == ref_norm

            if is_exact:
                exact_matches += 1

            details.append({
                'pred_raw': pred_raw[:500] if pred_raw else '',
                'pred_extracted': pred,
                'gold': ref,
                'exact_match': is_exact,
            })

        exact_rate = exact_matches / total * 100 if total > 0 else 0

        return {
            'exact_match': exact_rate,
            'exact_count': exact_matches,
            'total_count': total,
            'details': details,
        }

    def _score_smiles(self, predictions: List[str],
                     references: List[str]) -> Dict[str, Any]:
        """Score for SMILES prediction tasks (fs, retro, nepp).
        
        Uses evaluation metrics from official MoleculeSMILESEvaluator.
        """
        total = len(predictions)
        exact_matches = 0
        valid_count = 0
        
        morgan_sims = []
        maccs_sims = []
        rdk_sims = []
        levenshtein_dists = []
        bleu_refs = []
        bleu_preds = []
        
        details = []

        for pred_raw, ref in zip(predictions, references):
            pred_smiles = _extract_answer_from_response(pred_raw, self.subtask)
            
            # Convert to canonical SMILES
            pred_canonical = _convert_to_canonical_smiles(pred_smiles) if pred_smiles else None
            ref_canonical = _convert_to_canonical_smiles(ref) if ref else None

            # Check validity
            pred_valid = pred_canonical is not None
            if pred_valid:
                valid_count += 1

            # Exact match (using InChI)
            is_exact = False
            if pred_smiles and ref:
                is_exact = _exact_match_smiles(pred_smiles, ref)
            if is_exact:
                exact_matches += 1

            # Similarities (only if both are valid)
            morgan_sim = 0.0
            maccs_sim = 0.0
            rdk_sim = 0.0
            lev_dist = 0
            
            if pred_canonical and ref_canonical:
                morgan_sim = _calculate_morgan_similarity(pred_canonical, ref_canonical)
                maccs_sim = _calculate_maccs_similarity(pred_canonical, ref_canonical)
                rdk_sim = _calculate_rdk_similarity(pred_canonical, ref_canonical)
                lev_dist = _calculate_levenshtein(pred_canonical, ref_canonical)
                
                # Collect for BLEU calculation
                bleu_refs.append([[c for c in ref_canonical]])
                bleu_preds.append([c for c in pred_canonical])
            else:
                # Handle invalid predictions
                if ref_canonical:
                    bleu_refs.append([[c for c in ref_canonical]])
                    bleu_preds.append([])
            
            morgan_sims.append(morgan_sim)
            maccs_sims.append(maccs_sim)
            rdk_sims.append(rdk_sim)
            levenshtein_dists.append(lev_dist)

            details.append({
                'pred_raw': pred_raw[:500] if pred_raw else '',
                'pred_smiles': pred_smiles,
                'pred_canonical': pred_canonical,
                'gold': ref,
                'gold_canonical': ref_canonical,
                'pred_valid': pred_valid,
                'exact_match': is_exact,
                'morgan_similarity': morgan_sim,
                'maccs_similarity': maccs_sim,
                'rdk_similarity': rdk_sim,
                'levenshtein': lev_dist,
            })

        # Calculate BLEU
        bleu_score = 0.0
        if bleu_refs and bleu_preds:
            try:
                from nltk.translate.bleu_score import corpus_bleu
                bleu_score = corpus_bleu(bleu_refs, bleu_preds)
            except Exception:
                pass

        exact_match_rate = exact_matches / total * 100 if total > 0 else 0
        validity_rate = valid_count / total * 100 if total > 0 else 0
        avg_morgan = sum(morgan_sims) / total * 100 if total > 0 else 0
        avg_maccs = sum(maccs_sims) / total * 100 if total > 0 else 0
        avg_rdk = sum(rdk_sims) / total * 100 if total > 0 else 0
        avg_levenshtein = sum(levenshtein_dists) / total if total > 0 else 0

        return {
            'exact_match': exact_match_rate,
            'validity': validity_rate,
            'morgan_sims': avg_morgan,
            'maccs_sims': avg_maccs,
            'rdk_sims': avg_rdk,
            'bleu': bleu_score * 100,
            'levenshtein': avg_levenshtein,
            'exact_match_count': exact_matches,
            'valid_count': valid_count,
            'total_count': total,
            'details': details,
        }
