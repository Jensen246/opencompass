"""PANORAMA PAR4PC (Prior Art Retrieval) Dataset and Evaluator.

PAR4PC task: Given a patent claim and 8 candidate patents (A-H), identify which
patents were cited as prior art references for the claim rejection.

Reference: https://huggingface.co/datasets/LG-AI-Research/PANORAMA
"""

import json
import re
from typing import Any, Dict, List, Optional, Set

import numpy as np
from datasets import Dataset, load_dataset
from sklearn.metrics import f1_score

from opencompass.openicl.icl_evaluator import BaseEvaluator
from opencompass.registry import ICL_EVALUATORS, LOAD_DATASET

from ..base import BaseDataset


def _parse_json_field(raw_value: Any, default: Any = None) -> Any:
    """Parse a JSON string field, return default if parsing fails."""
    if isinstance(raw_value, str):
        try:
            return json.loads(raw_value)
        except json.JSONDecodeError:
            return default if default is not None else raw_value
    return raw_value if raw_value is not None else default


def _create_par4pc_prompt(item: Dict[str, Any], prompt_mode: str = 'zero-shot') -> str:
    """Create prompt for PAR4PC task.

    Args:
        item: A single data item from the dataset
        prompt_mode: 'zero-shot' or 'cot'

    Returns:
        Formatted prompt string
    """
    context = _parse_json_field(item.get('context', {}), {})
    options = _parse_json_field(item.get('options', {}), {})
    claim_number = item.get('claim_number', 'N/A')
    app_number = item.get('application_number', 'N/A')

    context_claims_data = context.get('claims', [])
    if isinstance(context_claims_data, np.ndarray):
        context_claims_data = context_claims_data.tolist()
    context_claims_json = json.dumps(context_claims_data, indent=4)

    prompt = f"""You are a patent expert tasked with identifying cited patents for a specific claim rejection based *only* on the provided context and options.

**Context:**
*   **Application Number:** {app_number}
*   **Title:** {context.get("title", "N/A")}
*   **Abstract:** {context.get("abstract", "N/A")}
*   **Initial Claims:**
    {context_claims_json}

**Target Claim for Analysis:** Claim {claim_number}

**Options (Potential Cited Patents):**
"""

    for letter, details in sorted(options.items()):
        if isinstance(details, str):
            try:
                details = json.loads(details)
            except json.JSONDecodeError:
                details = {}
        elif not isinstance(details, dict):
            details = {}

        prompt += f"\n{letter}: Patent ID: {details.get('patent_id', 'N/A')}\n"
        prompt += f"   Title: {details.get('title', 'N/A')}\n"
        prompt += f"   Abstract: {details.get('abstract', 'N/A')}\n"

        option_claims_data = details.get('claims', [])
        if isinstance(option_claims_data, np.ndarray):
            option_claims_data = option_claims_data.tolist()
        claims_str = json.dumps(option_claims_data, indent=4) if option_claims_data else '[]'

        prompt += f'   Claims: {claims_str}\n'

    if prompt_mode == 'cot':
        prompt += f"""
**Step-by-Step Instruction**
*Apply the Broadest Reasonable Interpretation (BRI) standard.*

1. **BRI Claim Charting**
   • Decompose Claim {claim_number} into numbered limitations [L1]-[Ln] and record element : function/relationship.

2. **Core Inventive Concept & Problem**
   • Summarise in ≤ 20 words the inventive concept + technical problem.

3. **Single-Reference Screening (§102)**
   • For each option (A-H) rate coverage:
     | Opt | Maps limits | Term/synonym | Field match | Score* |
     |-----|------------|---------------|-------------|--------|
     *Score: 0 = no key feature, 1 = partial, 2 = full anticipation.*

4. **Multi-Reference Analysis (§103)**
   a. Pick options with Score ≥ 1.
   b. Build coverage matrix to find smallest combo covering all limits.
   c. For each viable combo, supply a motivation-to-combine (same field, complementary function, predictable substitution, etc.).
   d. Rank: full coverage → clear motivation → earliest primary art.

5. **Consistency & Inherency Check**
   • Reject art that contradicts any limitation; accept inherent feature only if necessarily present.

6. **Output (JSON only)**
    Always write the "reason" **first** and then write the "answer".
   • "reason" MUST include:
     Step1 <claim focus>; Step2 <mapping & motivation> ; Step3 <§102 or §103>.
   • Keep "reason" ≤ 200 words.
   • "answer" = single letter **or** list of letters.

```json
{{"reason":"Step1 ... ; Step2 ... ; Step3 ...", "answer":"A"}}
```
If multiple patents are cited:
```json
{{"reason":"Step1 ... ; Step2 ... ; Step3 ...", "answer": ["A","C","F"]}}
```
"""
    else:  # zero-shot
        prompt += f"""
Based only on the provided context and options, which patent(s) (A-H) were cited?
Answer format (JSON only):
```json
{{"answer": "A"}}
```
If multiple patents are cited:
```json
{{"answer": ["A","C","F"]}}
```
"""
    return prompt


def _parse_answer_string(answer_string: Any) -> List[str]:
    """Parse answer string to list of letters A-H.

    Args:
        answer_string: Raw answer string (comma-separated letters) or numpy array

    Returns:
        Sorted list of unique letters A-H
    """
    if isinstance(answer_string, np.ndarray):
        answer_string = ','.join(str(x) for x in answer_string)

    if not answer_string:
        return []

    answer_string = str(answer_string).upper()
    letters = re.split(r'[,\s]+', answer_string)
    valid_letters = [c.strip() for c in letters if c.strip() and 'A' <= c.strip() <= 'H']
    return sorted(list(set(valid_letters)))


def _extract_answer_from_response(response: str) -> List[str]:
    """Extract answer letters from model response.

    Args:
        response: Raw model response string

    Returns:
        List of extracted letters (A-H)
    """
    if not response:
        return []

    response = response.strip()

    # Try to parse as JSON first
    try:
        # Handle markdown code blocks
        if '```json' in response:
            json_match = re.search(r'```json\s*(.*?)\s*```', response, re.DOTALL)
            if json_match:
                response = json_match.group(1)
        elif '```' in response:
            json_match = re.search(r'```\s*(.*?)\s*```', response, re.DOTALL)
            if json_match:
                response = json_match.group(1)

        data = json.loads(response)
        if isinstance(data, dict):
            ans = data.get('answer', [])

            if isinstance(ans, str):
                # Single letter or comma-separated
                if len(ans.strip()) == 1 and 'A' <= ans.strip().upper() <= 'H':
                    return [ans.strip().upper()]
                return _parse_answer_string(ans)

            if isinstance(ans, list):
                result = []
                for item in ans:
                    if isinstance(item, str) and len(item.strip()) == 1:
                        letter = item.strip().upper()
                        if 'A' <= letter <= 'H':
                            result.append(letter)
                return sorted(list(set(result)))

    except json.JSONDecodeError:
        pass

    # Fallback: regex extraction
    patterns = [
        r'"answer"\s*:\s*\[\s*"([A-H])"\s*(?:,\s*"([A-H])"\s*)*\]',
        r'"answer"\s*:\s*"([A-H](?:,\s*[A-H])*)"',
        r'"answer"\s*:\s*"([A-H])"',
    ]

    for pattern in patterns:
        match = re.search(pattern, response, re.IGNORECASE)
        if match:
            found = match.group(0)
            letters = re.findall(r'[A-H]', found, re.IGNORECASE)
            if letters:
                return sorted(list(set([l.upper() for l in letters])))

    # Last resort: find any A-H letters in the response
    letters = re.findall(r'\b([A-H])\b', response, re.IGNORECASE)
    if letters:
        return sorted(list(set([l.upper() for l in letters])))

    return []


@LOAD_DATASET.register_module()
class PAR4PCDataset(BaseDataset):
    """PANORAMA PAR4PC (Prior Art Retrieval) Dataset.

    This dataset loads the PAR4PC task from the PANORAMA benchmark, which
    involves identifying which patents (A-H) were cited as prior art.
    """

    @staticmethod
    def load(path: str = 'LG-AI-Research/PANORAMA',
             prompt_mode: str = 'zero-shot',
             max_input_len: Optional[int] = None,
             tokenizer_path: Optional[str] = None,
             **kwargs) -> Dataset:
        """Load and preprocess the PAR4PC dataset.

        Args:
            path: HuggingFace dataset path
            prompt_mode: 'zero-shot' or 'cot'
            max_input_len: Maximum input token length. Samples exceeding this
                will be filtered out. If None, no filtering is applied.
            tokenizer_path: Path to tokenizer for length calculation.
                Required if max_input_len is specified.

        Returns:
            Dataset with columns: prompt, gold_answers, silver_answers,
                                  negative_answers, identifier
        """
        ds = load_dataset(path, data_dir='PAR4PC', split='test')
        original_size = len(ds)

        # Load tokenizer if max_input_len is specified
        tokenizer = None
        if max_input_len is not None and tokenizer_path:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_path, trust_remote_code=True)

        def process_item(item: Dict[str, Any]) -> Dict[str, Any]:
            prompt = _create_par4pc_prompt(item, prompt_mode)

            # Filter by token length if tokenizer is available
            if tokenizer is not None and max_input_len is not None:
                tokens = tokenizer.encode(prompt, add_special_tokens=False)
                if len(tokens) > max_input_len:
                    return {
                        'prompt': None,
                        'gold_answers': None,
                        'silver_answers': None,
                        'negative_answers': None,
                        'identifier': None,
                    }

            return {
                'prompt': prompt,
                'gold_answers': item.get('gold_answers', ''),
                'silver_answers': item.get('silver_answers', ''),
                'negative_answers': item.get('negative_answers', ''),
                'identifier': f"par4pc_app{item.get('application_number', 'N/A')}_claim{item.get('claim_number', 'N/A')}",
            }

        processed_ds = ds.map(process_item, remove_columns=ds.column_names)

        # Filter out samples that exceeded max_input_len
        if tokenizer is not None and max_input_len is not None:
            processed_ds = processed_ds.filter(lambda x: x['prompt'] is not None)
            filtered_count = original_size - len(processed_ds)
            print(f'[PAR4PC] Filtered {filtered_count} samples exceeding '
                  f'{max_input_len} tokens. Remaining: {len(processed_ds)}/{original_size}')

        return processed_ds


@ICL_EVALUATORS.register_module()
class PAR4PCEvaluator(BaseEvaluator):
    """Evaluator for PANORAMA PAR4PC task.

    Computes:
    - Exact Match: Percentage of predictions that exactly match gold answers
    - Custom Score: (TP*2 - FP - FN) / (gold*2) * 100
    - Macro F1: Macro-averaged F1 score
    """

    def score(self, predictions: List[str], references: List[Dict[str, str]],
              test_set: Optional[Dataset] = None) -> Dict[str, Any]:
        """Compute evaluation metrics.

        Args:
            predictions: List of model predictions (raw response strings)
            references: List of dicts with gold_answers, silver_answers, negative_answers
            test_set: Optional test dataset

        Returns:
            Dictionary containing exact_match, custom_score, macro_f1, and details
        """
        if len(predictions) != len(references):
            return {'error': 'predictions and references have different length'}

        exact_matches = 0
        custom_scores = []
        all_tp = 0
        all_fp = 0
        all_fn = 0
        details = []

        for pred_raw, ref in zip(predictions, references):
            # Extract predictions
            pred_letters = _extract_answer_from_response(pred_raw)

            # Parse reference answers
            if isinstance(ref, dict):
                gold = _parse_answer_string(ref.get('gold_answers', ''))
                silver = _parse_answer_string(ref.get('silver_answers', ''))
                negative = _parse_answer_string(ref.get('negative_answers', ''))
            else:
                # Fallback: ref is just the gold_answers string
                gold = _parse_answer_string(ref)
                silver = []
                negative = []

            pred_set = set(pred_letters)
            gold_set = set(gold)
            silver_set = set(silver)
            negative_set = set(negative)

            # Exact match
            is_exact = pred_set == gold_set
            if is_exact:
                exact_matches += 1

            # Custom score calculation
            # TP = predictions that are in gold
            # FP = predictions that are not in gold or silver
            # FN = gold items that were not predicted
            tp = len(pred_set & gold_set)
            fp = len(pred_set - gold_set - silver_set)
            fn = len(gold_set - pred_set)

            all_tp += tp
            all_fp += fp
            all_fn += fn

            if gold_set:
                # Custom score: (TP*2 - FP - FN) / (gold*2) * 100
                raw_score = tp * 2 - fp - fn
                max_score = len(gold_set) * 2
                custom_score = max(0, raw_score / max_score * 100) if max_score > 0 else 0
            else:
                custom_score = 100 if not pred_set else 0

            custom_scores.append(custom_score)

            detail = {
                'pred_raw': pred_raw[:500] if pred_raw else '',
                'pred_extracted': pred_letters,
                'gold': gold,
                'silver': silver,
                'exact_match': is_exact,
                'custom_score': custom_score,
            }
            details.append(detail)

        # Aggregate metrics
        total = len(predictions)
        exact_match_rate = (exact_matches / total * 100) if total > 0 else 0
        avg_custom_score = (sum(custom_scores) / total) if total > 0 else 0

        # Macro F1 calculation
        # For multi-label, we compute precision, recall, F1
        if all_tp + all_fp > 0:
            precision = all_tp / (all_tp + all_fp)
        else:
            precision = 0

        if all_tp + all_fn > 0:
            recall = all_tp / (all_tp + all_fn)
        else:
            recall = 0

        if precision + recall > 0:
            macro_f1 = 2 * precision * recall / (precision + recall) * 100
        else:
            macro_f1 = 0

        return {
            'exact_match': exact_match_rate,
            'custom_score': avg_custom_score,
            'macro_f1': macro_f1,
            'precision': precision * 100,
            'recall': recall * 100,
            'total_count': total,
            'details': details,
        }
