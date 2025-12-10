"""PANORAMA NOC4PC (Novelty/Obviousness Classification) Dataset and Evaluator.

NOC4PC task: Given a patent claim and prior art references, predict whether the claim
should be ALLOWED, rejected under 35 U.S.C. § 102 (lack of novelty), or rejected
under 35 U.S.C. § 103 (obviousness).

Reference: https://huggingface.co/datasets/LG-AI-Research/PANORAMA
"""

import json
import re
from typing import Any, Dict, List, Optional

from datasets import Dataset, load_dataset
from sklearn.metrics import accuracy_score, f1_score

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


def _create_noc4pc_prompt(item: Dict[str, Any], prompt_mode: str = 'zero-shot') -> str:
    """Create prompt for NOC4PC task.

    Args:
        item: A single data item from the dataset
        prompt_mode: 'zero-shot' or 'cot'

    Returns:
        Formatted prompt string
    """
    app_num = item.get('application_number', 'N/A')
    claim_num = item.get('claim_number', 'N/A')

    context = _parse_json_field(item.get('context', {}), {})
    prior_art_specs = _parse_json_field(item.get('prior_art_specifications', []), [])

    target_title = context.get('title', 'N/A')
    target_abstract = context.get('abstract', 'N/A')
    target_claims = context.get('claims', [])

    # Extract target claim text
    target_claim_text = 'N/A'
    try:
        claim_idx = int(claim_num) - 1
        if 0 <= claim_idx < len(target_claims):
            claim_item = target_claims[claim_idx]
            if isinstance(claim_item, dict) and 'claim_text' in claim_item:
                target_claim_text = str(claim_item['claim_text'])
            elif isinstance(claim_item, str):
                target_claim_text = claim_item
            else:
                target_claim_text = str(claim_item)
    except (ValueError, TypeError):
        pass

    prompt = f"""You are an expert AI acting as a U.S. Patent Examiner.
Your task is to analyze **Target Claim {claim_num}** of the **Target Patent Application** in view of the provided **Prior Art Specifications**.

Determine if **Target Claim {claim_num}** is allowable or should be rejected under 35 U.S.C. § 102 (lack of novelty) or 35 U.S.C. § 103 (obviousness).

**Target Patent Application Information:**
*   Application Number: {app_num}
*   Target Claim Number: {claim_num}
*   Title: {target_title}
*   Abstract: {target_abstract}
*   Target Claim {claim_num} Text to Analyze:
    ```
    {target_claim_text}
    ```

**Prior Art Specifications (Cited as Basis for Potential Rejection):**
The following prior art documents and specific paragraphs were cited as potentially relevant for the rejection of the target claim. Analyze the target claim against the information presented in these specific paragraphs **and the claims** of the prior art.
"""

    if not prior_art_specs:
        prompt += 'No prior art specifications provided for analysis.\n'
    else:
        for i, spec in enumerate(prior_art_specs):
            prompt += f'\n--- Prior Art #{i+1} ---\n'
            prompt += f"*   Patent ID: {spec.get('patent_id', 'N/A')}\n"
            prompt += f"*   Title: {spec.get('title', 'N/A')}\n"
            prompt += f"*   Abstract: {spec.get('abstract', 'N/A')}\n"

            prior_art_claims = spec.get('claims', [])
            prompt += '*   Claims:\n'
            if not prior_art_claims:
                prompt += '    (No claims provided for this prior art)\n'
            else:
                for idx, claim_content in enumerate(prior_art_claims):
                    claim_text_display = str(claim_content)
                    if isinstance(claim_content, dict) and 'claim_text' in claim_content:
                        claim_text_display = str(claim_content['claim_text'])

                    claim_lines = claim_text_display.splitlines()
                    if claim_lines:
                        prompt += f'    {idx+1}. {claim_lines[0]}\n'
                        for line in claim_lines[1:]:
                            prompt += f'       {line}\n'
                    else:
                        prompt += f'    {idx+1}.\n'

            prompt += '*   Cited Paragraphs from Specification (Basis for Analysis):\n'
            paragraphs = spec.get('paragraphs', [])
            if not paragraphs:
                prompt += '    (No specific paragraphs cited for this prior art)\n'
            else:
                for para in paragraphs:
                    key_display = para.get('key', 'N/A')
                    content_display = para.get('content', '')
                    content_lines = content_display.splitlines()
                    if content_lines:
                        prompt += f'    [{key_display}]: {content_lines[0]}\n'
                        for line in content_lines[1:]:
                            prompt += f'          {line}\n'
                    else:
                        prompt += f'    [{key_display}]:\n'

    if prompt_mode == 'cot':
        prompt += f"""
**Analysis Task and Response Instructions:**

Perform your internal reasoning first, then draft the *Office-Action-style* text.

---

### INTERNAL REASONING (not shown to applicant)
1. Apply the Broadest-Reasonable-Interpretation (BRI) to Claim {claim_num}; chart every limitation [L1]-[Ln].
   1-a. *Statutory check* - confirm Claim{claim_num} fits a statutory class (process, machine, manufacture, composition).
   1-b. *Limitation numbering*-  break the claim into [L1]-[Ln] and record in "element : function / relationship" form.
   1-c. *Key-feature flag* - mark limitations asserted (or apparent) as novel / non-obvious.
2. Compare each limitation to the teachings (claims + cited paragraphs) of every prior-art reference.
3. Decide:
   • §102 anticipation if a single reference explicitly, implicitly, or inherently discloses each and every limitation.
        Under BRI, interpret broadly: functional equivalence or conventional components (processors, databases, modules, memory, standard network elements, known protocols, etc.) count as implicit disclosures.
    • §103 obviousness if any of the following apply:
        (a) A primary reference discloses at least 70% of the limitations explicitly or implicitly, and remaining limitations constitute routine modifications, predictable optimizations (e.g., efficiency, speed, cost reduction, miniaturization), or standard practices known to a person of ordinary skill in the field.
        (b) A combination of references collectively covers all limitations and demonstrates a clear, implicit or explicit KSR rationale, such as addressing the same technical problem, improving performance, enhancing usability, or following common industry practices.
        (c) The limitations not explicitly disclosed are obvious through common general knowledge or widely recognized industry standards or textbooks in the field.
    • ALLOW Only if:
    No single reference or combination of references, even considering implicit disclosures and general knowledge, discloses or renders obvious specific, detailed implementation aspects (unique structures, algorithmic specifics, or non-trivial process steps), AND
    No reasonable motivation or rationale (performance improvement, standard practice, or known solution) can be objectively articulated to bridge these gaps.

---

### DRAFT OA LANGUAGE (will be revealed)
Write the *reason* paragraph exactly like an Office Action:

* Start: **"Regarding Claim {claim_num}, ..."**
* Use examiner diction:
  - "Reference X (Col. Y, lines Z) discloses ..."
  - "Therefore, Claim {claim_num} is rejected under 35 U.S.C. § 102(a) as being anticipated by Reference X."
  - or "It would have been obvious to one of ordinary skill to modify X with Y (same field, predictable results) ... ⇒ § 103 rejection."
  - or "The cited references do not teach or render obvious limitation [Lk] ... Claim {claim_num} is allowable."

* If § 103, list **all** references in the combination (e.g., "in view of Smith '123").
* Cite at least one column-line or paragraph for each matched limitation.
* Keep length ≤ 200 words.

---

### OUTPUT (JSON only)
Always write the "reason" first and then write the "answer".
Return exactly one JSON object:

```json
{{
  "reason": "<OA-style paragraph above>",
  "code": "102" | "103" | "ALLOW"
}}
```
"""
    else:  # zero-shot
        prompt += f"""
**Select Conclusion Code**
Choose one: "ALLOW", "102", or "103".
    *   `"ALLOW"`: If your reasoning concluded the claim is novel and non-obvious over the cited art.
    *   `"102"` (Rejected - Novelty): If your reasoning concluded the claim is anticipated by a single cited reference.
    *   `"103"` (Rejected - Obviousness): If your reasoning concluded the claim is obvious over the cited art.

**Answer Format (JSON only)**
Return ONLY this JSON object - DO NOT INCLUDE ANY REASON IN THE ANSWER.

```json
{{"code": "102"}}
```
"""
    return prompt


def _extract_code_from_response(response: str) -> Optional[str]:
    """Extract rejection code from model response.

    Args:
        response: Raw model response string

    Returns:
        Extracted code ('102', '103', 'ALLOW') or None if extraction fails
    """
    if not response:
        return None

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
            code = data.get('code')
            if code:
                code = str(code).strip().upper()
                if code in {'ALLOW', '102', '103'}:
                    return code
    except json.JSONDecodeError:
        pass

    # Fallback: regex extraction
    patterns = [
        r'"code"\s*:\s*"(ALLOW|102|103)"',
        r'"code"\s*:\s*\'(ALLOW|102|103)\'',
        r'\bcode\s*[:=]\s*(ALLOW|102|103)\b',
    ]

    for pattern in patterns:
        match = re.search(pattern, response, re.IGNORECASE)
        if match:
            code = match.group(1).upper()
            if code in {'ALLOW', '102', '103'}:
                return code

    # Last resort: look for standalone codes
    for code in ['ALLOW', '102', '103']:
        if code in response.upper():
            return code

    return None


@LOAD_DATASET.register_module()
class NOC4PCDataset(BaseDataset):
    """PANORAMA NOC4PC (Novelty/Obviousness Classification) Dataset.

    This dataset loads the NOC4PC task from the PANORAMA benchmark, which
    involves classifying patent claims as ALLOW, 102 (novelty rejection),
    or 103 (obviousness rejection).
    """

    @staticmethod
    def load(path: str = 'LG-AI-Research/PANORAMA',
             prompt_mode: str = 'zero-shot',
             **kwargs) -> Dataset:
        """Load and preprocess the NOC4PC dataset.

        Args:
            path: HuggingFace dataset path
            prompt_mode: 'zero-shot' or 'cot'

        Returns:
            Dataset with columns: prompt, gold_code, identifier
        """
        ds = load_dataset(path, data_dir='NOC4PC', split='test')

        def process_item(item: Dict[str, Any]) -> Dict[str, Any]:
            prompt = _create_noc4pc_prompt(item, prompt_mode)

            # Extract gold answer
            answer_raw = item.get('answer', {})
            answer = _parse_json_field(answer_raw, {})
            gold_code = str(answer.get('code', '')).strip().upper() if answer.get('code') else ''

            return {
                'prompt': prompt,
                'gold_code': gold_code,
                'identifier': f"noc4pc_app{item.get('application_number', 'N/A')}_claim{item.get('claim_number', 'N/A')}",
            }

        processed_ds = ds.map(process_item, remove_columns=ds.column_names)
        return processed_ds


@ICL_EVALUATORS.register_module()
class NOC4PCEvaluator(BaseEvaluator):
    """Evaluator for PANORAMA NOC4PC task.

    Computes:
    - Accuracy: Overall classification accuracy
    - Macro F1: Macro-averaged F1 score across all classes
    """

    def score(self, predictions: List[str], references: List[str],
              test_set: Optional[Dataset] = None) -> Dict[str, Any]:
        """Compute evaluation metrics.

        Args:
            predictions: List of model predictions (raw response strings)
            references: List of gold labels (code strings)
            test_set: Optional test dataset (not used)

        Returns:
            Dictionary containing accuracy, macro_f1, and details
        """
        if len(predictions) != len(references):
            return {'error': 'predictions and references have different length'}

        # Extract codes from predictions
        extracted_preds = []
        details = []

        for pred, ref in zip(predictions, references):
            extracted = _extract_code_from_response(pred)
            extracted_preds.append(extracted)

            detail = {
                'pred_raw': pred[:500] if pred else '',  # Truncate for readability
                'pred_extracted': extracted,
                'gold': ref,
                'correct': extracted is not None and extracted == ref.upper(),
            }
            details.append(detail)

        # Filter out None predictions for metric calculation
        valid_pairs = [(p, r) for p, r in zip(extracted_preds, references)
                       if p is not None and r]

        if not valid_pairs:
            return {
                'accuracy': 0.0,
                'macro_f1': 0.0,
                'valid_count': 0,
                'total_count': len(predictions),
                'details': details,
            }

        valid_preds, valid_refs = zip(*valid_pairs)

        # Normalize references
        valid_refs = [r.upper() for r in valid_refs]

        # Calculate metrics
        accuracy = accuracy_score(valid_refs, valid_preds) * 100
        macro_f1 = f1_score(valid_refs, valid_preds,
                           labels=['102', '103', 'ALLOW'],
                           average='macro',
                           zero_division=0) * 100

        return {
            'accuracy': accuracy,
            'macro_f1': macro_f1,
            'valid_count': len(valid_pairs),
            'total_count': len(predictions),
            'details': details,
        }
