"""ChemCoTBench Molecule Optimization Dataset and Evaluator.

mol_opt tasks:
- Target-level: drd, gsk, jnk (drug target optimization)
- Physicochemical-level: logp, qed, solubility (property optimization)

Reference: https://github.com/IDEA-XL/ChemCoTBench
           https://huggingface.co/datasets/OpenMol/ChemCoTBench
"""

import json
import re
from collections import namedtuple
from typing import Any, Dict, List, Optional

from datasets import Dataset, load_dataset

from opencompass.openicl.icl_evaluator import BaseEvaluator
from opencompass.registry import ICL_EVALUATORS, LOAD_DATASET

from ..base import BaseDataset


# Property name mapping to match official implementation
PROP_NAME_MAP = {
    'drd': 'drd2',
    'jnk': 'jnk3',
    'gsk': 'gsk3b',
    'logp': 'logp',
    'qed': 'qed',
    'solubility': 'solubility',
}


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
            # Handle "Final Target Molecule" field (official format)
            answer = (data.get('Final Target Molecule') or 
                     data.get('Final_Target_Molecule') or
                     data.get('output') or 
                     data.get('answer'))
            if answer is not None:
                return str(answer).strip()
    except json.JSONDecodeError:
        pass

    # Fallback: regex extraction
    patterns = [
        r'"Final Target Molecule"\s*:\s*"([^"]*)"',
        r'"Final_Target_Molecule"\s*:\s*"([^"]*)"',
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


def _extract_source_smiles(query: str) -> Optional[str]:
    """Extract source molecule SMILES from the query."""
    match = re.search(r'Source Molecule:\s*([^\s.]+)', query)
    if match:
        return match.group(1).strip()
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


class ESOLCalculator:
    """ESOL solubility calculator from official implementation.
    
    Reference: https://github.com/IDEA-XL/ChemCoTBench/blob/main/baseline_and_eval/eval/eval_metric.py
    """
    
    def __init__(self):
        from rdkit import Chem
        self.aromatic_query = Chem.MolFromSmarts("a")
        self.Descriptor = namedtuple("Descriptor", "mw logp rotors ap")

    def calc_ap(self, mol):
        """Calculate aromatic proportion."""
        matches = mol.GetSubstructMatches(self.aromatic_query)
        return len(matches) / mol.GetNumAtoms() if mol.GetNumAtoms() > 0 else 0

    def calc_esol_descriptors(self, mol):
        """Calculate mw, logp, rotors and aromatic proportion."""
        from rdkit.Chem import Descriptors, Crippen, Lipinski
        
        mw = Descriptors.MolWt(mol)
        logp = Crippen.MolLogP(mol)
        rotors = Lipinski.NumRotatableBonds(mol)
        ap = self.calc_ap(mol)
        return self.Descriptor(mw=mw, logp=logp, rotors=rotors, ap=ap)

    def calc_esol(self, smiles: str) -> Optional[float]:
        """Calculate ESOL solubility."""
        from rdkit import Chem
        
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        
        try:
            # Coefficients from official implementation
            intercept = 0.26121066137801696
            coef = {
                'mw': -0.0066138847738667125,
                'logp': -0.7416739523408995,
                'rotors': 0.003451545565957996,
                'ap': -0.42624840441316975
            }
            desc = self.calc_esol_descriptors(mol)
            esol = (intercept + 
                   coef['logp'] * desc.logp + 
                   coef['mw'] * desc.mw + 
                   coef['rotors'] * desc.rotors + 
                   coef['ap'] * desc.ap)
            return esol
        except Exception:
            return None


class PropertyOracle:
    """Property oracle using TDC for drug targets and RDKit for physicochemical properties.
    
    Reference: https://github.com/IDEA-XL/ChemCoTBench/blob/main/baseline_and_eval/eval/eval_metric.py
    """
    
    def __init__(self, prop: str):
        self.prop = prop
        self.oracle = None
        self._init_oracle()
    
    def _init_oracle(self):
        """Initialize the appropriate oracle based on property type."""
        prop_mapped = PROP_NAME_MAP.get(self.prop, self.prop)
        
        if prop_mapped in ['drd2', 'jnk3', 'gsk3b', 'logp', 'qed']:
            # Use TDC Oracle for these properties
            try:
                from tdc import Oracle
                self.oracle = Oracle(name=prop_mapped)
            except ImportError:
                # Fallback to RDKit for logp and qed if TDC not available
                if prop_mapped in ['logp', 'qed']:
                    self.oracle = self._get_rdkit_oracle(prop_mapped)
                else:
                    self.oracle = None
            except Exception:
                self.oracle = None
        elif prop_mapped == 'solubility':
            try:
                self.oracle = ESOLCalculator().calc_esol
            except Exception:
                self.oracle = None
    
    def _get_rdkit_oracle(self, prop: str):
        """Get RDKit-based oracle as fallback."""
        def rdkit_oracle(smiles: str) -> Optional[float]:
            try:
                from rdkit import Chem
                from rdkit.Chem import Descriptors
                
                mol = Chem.MolFromSmiles(smiles)
                if mol is None:
                    return None
                
                if prop == 'logp':
                    return Descriptors.MolLogP(mol)
                elif prop == 'qed':
                    return Descriptors.qed(mol)
                return None
            except Exception:
                return None
        return rdkit_oracle
    
    def __call__(self, smiles: str) -> Optional[float]:
        """Calculate property value for given SMILES."""
        if self.oracle is None:
            return None
        try:
            return self.oracle(smiles)
        except Exception:
            return None


def _calculate_scaffold_similarity(smiles1: str, smiles2: str) -> tuple:
    """Calculate scaffold consistency (hard and soft).
    
    Based on official implementation using rdFMCS.FindMCS.
    Reference: https://github.com/IDEA-XL/ChemCoTBench/blob/main/baseline_and_eval/eval/eval_metric.py
    
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
        
        # Hard: exact match
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
    except Exception:
        return False, 0.0


def _compute_statistics(numbers: List[float], prop: str, skew: bool = True) -> Dict[str, float]:
    """Compute statistics with winsorization.
    
    Based on official implementation.
    """
    import numpy as np
    
    if not numbers:
        return {
            'mean': 0.0,
            'variance': 0.0,
            'min': 0.0,
            'max': 0.0,
            'success_rate': 0.0,
            'best_rate': 0.0,
        }
    
    # Thresholds from official implementation
    threshold_dict = {
        'gsk3b': 0.3, 'gsk': 0.3,
        'qed': 0.3,
        'drd2': 0.3, 'drd': 0.3,
        'jnk3': 0.3, 'jnk': 0.3,
        'logp': 0.5,
        'solubility': 0.5,
    }
    
    numbers_arr = np.array(numbers)
    
    if skew:
        # Winsorize at 5th and 95th percentiles
        lower = np.percentile(numbers_arr, 5)
        upper = np.percentile(numbers_arr, 95)
        winsorized = np.clip(numbers_arr, lower, upper)
        mean = float(np.mean(winsorized))
        variance = float(np.var(winsorized))
        min_val = float(np.min(winsorized))
        max_val = float(np.max(winsorized))
    else:
        mean = float(np.mean(numbers_arr))
        variance = float(np.var(numbers_arr))
        min_val = float(np.min(numbers_arr))
        max_val = float(np.max(numbers_arr))
    
    success_rate = sum(1 for x in numbers if x > 0) / len(numbers)
    threshold = threshold_dict.get(prop, 0.5)
    best_rate = sum(1 for x in numbers if x >= threshold) / len(numbers)
    
    return {
        'mean': mean,
        'variance': variance,
        'min': min_val,
        'max': max_val,
        'success_rate': success_rate,
        'best_rate': best_rate,
    }


@LOAD_DATASET.register_module()
class ChemCoTBenchMolOptDataset(BaseDataset):
    """ChemCoTBench Molecule Optimization Dataset."""

    @staticmethod
    def load(path: str = 'OpenMol/ChemCoTBench',
             subtask: str = 'logp',
             seed: int = 42,
             **kwargs) -> Dataset:
        """Load and preprocess the mol_opt dataset from HuggingFace."""
        ds = load_dataset(path, split='train', trust_remote_code=True)
        ds = ds.filter(lambda x: x['task'] == 'mol_opt' and x['subtask'] == subtask)

        def process_item(item: Dict[str, Any]) -> Dict[str, Any]:
            # Extract source SMILES from query or meta
            source_smiles = _extract_source_smiles(item['query'])
            
            meta = item.get('meta', '{}')
            if isinstance(meta, str):
                try:
                    meta = json.loads(meta)
                except json.JSONDecodeError:
                    meta = {}
            
            if not source_smiles:
                source_smiles = meta.get('source', meta.get('src', ''))

            return {
                'prompt': item['query'],
                'source_smiles': source_smiles,
                'answer': source_smiles,  # For reference, use source as baseline
                'identifier': item.get('id', ''),
                'subtask': subtask,
            }

        processed_ds = ds.map(process_item, remove_columns=ds.column_names)
        processed_ds = processed_ds.shuffle(seed=seed)
        return processed_ds


@ICL_EVALUATORS.register_module()
class ChemCoTBenchMolOptEvaluator(BaseEvaluator):
    """Evaluator for ChemCoTBench mol_opt task.

    Evaluation based on official implementation:
    https://github.com/IDEA-XL/ChemCoTBench/blob/main/baseline_and_eval/eval/eval_molopt.py
    
    - property_improvement: Mean improvement in target property
    - success_rate: Rate of successful property improvement (> 0)
    - best_rate: Rate of improvement >= threshold
    - scaffold_consistency: Hard (exact) and soft (similarity) scaffold match
    """

    def __init__(self, subtask: str = 'logp') -> None:
        super().__init__()
        self.subtask = subtask
        self._oracle = None
    
    @property
    def oracle(self):
        """Lazy initialization of property oracle."""
        if self._oracle is None:
            self._oracle = PropertyOracle(self.subtask)
        return self._oracle

    def score(self, predictions: List[str], references: List[str],
              test_set: Optional[Dataset] = None) -> Dict[str, Any]:
        """Compute evaluation metrics."""
        if len(predictions) != len(references):
            return {'error': 'predictions and references have different length'}

        total = len(predictions)
        valid_smiles_count = 0
        valid_score_count = 0
        improvements = []
        scaffold_hard_count = 0
        scaffold_soft_sum = 0.0
        details = []

        # Get source SMILES
        source_list = references  # In mol_opt, references are source SMILES
        if test_set is not None:
            source_list = [item.get('source_smiles', '') for item in test_set]

        for i, (pred_raw, source) in enumerate(zip(predictions, source_list)):
            pred_smiles = _extract_smiles_from_response(pred_raw)
            
            # Check validity
            pred_valid = _is_valid_smiles(pred_smiles) if pred_smiles else False
            src_valid = _is_valid_smiles(source) if source else False
            
            if pred_valid:
                valid_smiles_count += 1
            
            # Calculate property improvement using oracle
            improvement = None
            src_prop = None
            tgt_prop = None
            
            if pred_valid and src_valid:
                src_prop = self.oracle(source)
                tgt_prop = self.oracle(pred_smiles)
                
                if src_prop is not None and tgt_prop is not None:
                    improvement = tgt_prop - src_prop
                    improvements.append(improvement)
                    valid_score_count += 1
            
            # Calculate scaffold consistency
            is_same_scaffold = False
            scaffold_sim = 0.0
            if pred_valid and src_valid:
                is_same_scaffold, scaffold_sim = _calculate_scaffold_similarity(source, pred_smiles)
                if is_same_scaffold:
                    scaffold_hard_count += 1
                scaffold_soft_sum += scaffold_sim

            # RDAgent compatibility: correct = valid SMILES with positive improvement
            is_correct = pred_valid and improvement is not None and improvement > 0

            details.append({
                'pred_raw': pred_raw[:500] if pred_raw else '',
                'pred_smiles': pred_smiles,
                'source_smiles': source,
                'pred_valid': pred_valid,
                'src_property': src_prop,
                'tgt_property': tgt_prop,
                'improvement': improvement,
                'scaffold_same': is_same_scaffold,
                'scaffold_similarity': scaffold_sim,
                'correct': is_correct,
            })

        # Calculate statistics
        valid_smiles_rate = valid_smiles_count / total * 100 if total > 0 else 0
        valid_score_rate = valid_score_count / total * 100 if total > 0 else 0
        
        # Improvement statistics (with winsorization for outliers)
        # Pad with zeros for invalid cases (as in official implementation)
        improvements_padded = improvements + [0.0] * (total - len(improvements))
        stats = _compute_statistics(improvements_padded, self.subtask, skew=True)
        
        scaffold_hard_rate = scaffold_hard_count / total * 100 if total > 0 else 0
        scaffold_soft_rate = scaffold_soft_sum / total * 100 if total > 0 else 0

        return {
            'mean_improvement': stats['mean'],
            'success_rate': stats['success_rate'] * 100,
            'best_rate': stats['best_rate'] * 100,
            'valid_smiles_rate': valid_smiles_rate,
            'valid_score_rate': valid_score_rate,
            'scaffold_hard': scaffold_hard_rate,
            'scaffold_soft': scaffold_soft_rate,
            'total_count': total,
            'details': details,
        }
