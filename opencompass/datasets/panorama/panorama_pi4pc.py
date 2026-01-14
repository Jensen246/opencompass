"""PANORAMA PI4PC (Paragraph Identification) Dataset and Evaluator.

PI4PC task: Given a patent claim and 5 candidate paragraphs from prior art,
identify the single most relevant paragraph that supports the claim rejection.

Reference: https://huggingface.co/datasets/LG-AI-Research/PANORAMA
"""

import json
import re
from typing import Any, Dict, List, Optional

from datasets import Dataset, load_dataset

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


def _create_pi4pc_prompt(item: Dict[str, Any], prompt_mode: str = 'zero-shot') -> str:
    """Create prompt for PI4PC task.

    Args:
        item: A single data item from the dataset
        prompt_mode: 'zero-shot' or 'cot'

    Returns:
        Formatted prompt string
    """
    context = _parse_json_field(item.get('context', {}), {})
    prior_art = _parse_json_field(item.get('prior_art_specification', {}), {})
    options = _parse_json_field(item.get('options', {}), {})

    claim_num = item.get('claim_number', 'N/A')
    app_num = item.get('application_number', 'N/A')

    # Extract target claim text
    target_claim_text = 'N/A'
    if 'claims' in context and isinstance(context['claims'], list):
        for c in context['claims']:
            if isinstance(c, dict) and c.get('claimNumber') == claim_num:
                target_claim_text = c.get('claim_text', 'N/A')
                break
        if target_claim_text == 'N/A' and str(claim_num).isdigit():
            claim_index = int(claim_num) - 1
            if 0 <= claim_index < len(context['claims']):
                claim_item = context['claims'][claim_index]
                if isinstance(claim_item, str):
                    target_claim_text = claim_item
                elif isinstance(claim_item, dict):
                    target_claim_text = claim_item.get('claim_text', 'N/A')

    # Format options
    options_text_list = []
    try:
        sorted_options = sorted(options.items(), key=lambda x: int(str(x[0])))
    except (ValueError, TypeError):
        sorted_options = list(options.items())
    for key, text in sorted_options:
        options_text_list.append(f'{key}: {text}')
    options_text = '\n'.join(options_text_list)

    # Base prompt template
    prompt = f"""You are an expert patent examiner reviewing a patent application.
Your task is to identify the **single most relevant paragraph** from the provided Prior Art Specification that is cited to reject Claim {claim_num} of the Target Application ({app_num}).

**Target Application Context:**
* **Title:** {context.get("title", "N/A")}
* **Abstract:** {context.get("abstract", "N/A")}
* **Claim {claim_num}:** {target_claim_text}

**Prior Art Specification Context:**
* **Patent ID:** {prior_art.get("patent_id", "N/A")}
* **Title:** {prior_art.get("title", "N/A")}
* **Abstract:** {prior_art.get("abstract", "N/A")}
* **Full Specification Text:**
{prior_art.get("specification", "Full specification text not available.")}

* **Cited Paragraph Options (Excerpts from the Full Specification above):**
Review the following paragraphs, which are excerpts from the full specification provided above. Choose the **one paragraph key (integer)** from the list below that best supports the rejection of Claim {claim_num}.

Options:
{options_text}

**CRITICAL INSTRUCTION:**
Based on your analysis of the **Full Specification Text** and the **Target Application Claim {claim_num}**, you **MUST** select **EXACTLY ONE** integer key from the **5 options** provided above.
Under no circumstances should you choose a key not present in the options or provide multiple keys, ranges, reasoning, or explanations.
"""

    if prompt_mode == 'cot':
        prompt += f"""
**Step-by-Step Method**
*Use the Broadest-Reasonable-Interpretation (BRI) standard throughout.*

1.  **BRI Claim Deconstruction**
    • Break the claim {claim_num} into **numbered limitations** (e.g., [1A]-[1F]).
    • Write each limitation in examiner-style "element : function / relationship" form.
    • Try to include as much of the claim as possible.

2.  **Key Distinguishing Feature(s)**
    • Identify which limitation(s) the applicant asserts as novel / non-obvious.
    • List all the features that should be considered when evaluating novelty and non-obviousness.

3.  **Prior-Art Mapping Table (one table per option paragraph)**
    • For each of the five option paragraphs, provide a detailed mapping to Claim {claim_num} elements.
    • Use the table format to score the degree of overlap between each option paragraph and the claim limitations.
    • IMPORTANT: **Do not skip any options** — evaluate all five paragraphs.
    | Opt# | Maps to elements | Exact term / BRI synonym | Col-Line (or ¶) | Match score* |
    |------|-----------------|--------------------------|-----------------|-------------|
    |  ##  |                 |                          |                 |             |
    |  ##  |                 |                          |                 |             |
    |  ##  |                 |                          |                 |             |
    |  ##  |                 |                          |                 |             |
    |  ##  |                 |                          |                 |             |
    *Scoring: 0 = missing, 1 = mention, 2 = partial, 3 = full & explicit.*

4.  **Select the Most Relevant Paragraph for Patentability Evaluation**
    • Your goal is to identify exactly ONE paragraph most relevant to evaluate the novelty/non-obviousness of the applicant's claimed invention.
    • Select exactly one paragraph based on its relevance to the novelty or non-obviousness of the Key Distinguishing Features (KD-x) in Claim {claim_num}.
    • Do not select multiple keys or provide general reasoning.
    • Focus on technical relevance, improvements, and system integration when selecting your paragraph.
    Selection Criteria:
    • Consider paragraphs that scored ≥ 1 points in Step 3.
    • Technical Objectives: Does the paragraph directly support the technical objectives of the Claim? Does it provide a solution to the problem presented by the Claim?
    • Prior Art Improvements: Does the paragraph present innovative improvements to existing systems or technologies?
    • System Integration: Does the paragraph explain how elements of the system described in the Claim interact or integrate with each other?
    • Motivation to Combine: Does the paragraph offer a motivational context for combining features, particularly for a §103 rejection?

    Output Requirements:
    • Clearly indicate your final selection as the Primary Reference (PR).
    • Provide a concise reason for your selection based strictly on the criteria above.

5. **Consistency & Inherency Check**
   • Verify the selected paragraph does not contradict any claim limitation.

6. **Output (JSON only)**
    Always write the "reason" first and then write the "answer".
   • 'reason' MUST list Step1-Step6 in order, each separated by ';'.
     ▸ Step1 <statutory/context> ;
     ▸ Step2 <limits> ;
     ▸ Step3 <key feature> ;
     ▸ Step4 <mapping & score> ;
     ▸ Step5 <rank/tie-break> ;
     ▸ Step6 <consistency/inherency & §102 or §103 result>.
   • Keep "reason" ≤ 1000 words.
   • "answer" = single paragraph key (int).

```json
{{"reason":"Step1 ... ; Step2 ... ; Step3 ... ; Step4 ...; Step5 ...; Step6 ...","answer": 17}}
```
"""
    else:  # zero-shot
        prompt += """
Answer format (JSON only)
Return ONLY this JSON object - DO NOT INCLUDE ANY REASON IN THE ANSWER.
```json
{"answer": ##}
```
"""
    return prompt


def _parse_answer_keys(answer_string: Any) -> List[int]:
    """Parse answer string to list of integer keys.

    Args:
        answer_string: Raw answer string or list

    Returns:
        List of integer keys
    """
    if not answer_string:
        return []

    if isinstance(answer_string, (int, float)):
        return [int(answer_string)]

    if isinstance(answer_string, str):
        # Try to extract integers from string
        numbers = re.findall(r'\d+', answer_string)
        return [int(n) for n in numbers]

    if isinstance(answer_string, list):
        result = []
        for item in answer_string:
            if isinstance(item, (int, float)):
                result.append(int(item))
            elif isinstance(item, str):
                try:
                    result.append(int(item))
                except ValueError:
                    pass
        return result

    return []


def _extract_answer_from_response(response: str) -> Optional[int]:
    """Extract answer integer from model response.

    Args:
        response: Raw model response string

    Returns:
        Extracted integer key or None if extraction fails
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
            ans = data.get('answer')
            if ans is not None:
                return int(ans)

    except (json.JSONDecodeError, ValueError, TypeError):
        pass

    # Fallback: regex extraction
    patterns = [
        r'"answer"\s*:\s*(\d+)',
        r'"answer"\s*:\s*"(\d+)"',
        r'answer\s*[:=]\s*(\d+)',
    ]

    for pattern in patterns:
        match = re.search(pattern, response, re.IGNORECASE)
        if match:
            try:
                return int(match.group(1))
            except ValueError:
                pass

    # Last resort: find any number in the response
    numbers = re.findall(r'\b(\d+)\b', response)
    if numbers:
        # Return the first number found
        return int(numbers[0])

    return None


@LOAD_DATASET.register_module()
class PI4PCDataset(BaseDataset):
    """PANORAMA PI4PC (Paragraph Identification) Dataset.

    This dataset loads the PI4PC task from the PANORAMA benchmark, which
    involves identifying the most relevant paragraph from prior art.
    """

    @staticmethod
    def load(path: str = 'LG-AI-Research/PANORAMA',
             prompt_mode: str = 'zero-shot',
             max_input_len: Optional[int] = None,
             tokenizer_path: Optional[str] = None,
             seed: int = 42,
             **kwargs) -> Dataset:
        """Load and preprocess the PI4PC dataset.

        Args:
            path: HuggingFace dataset path
            prompt_mode: 'zero-shot' or 'cot'
            max_input_len: Maximum input token length. Samples exceeding this
                will be filtered out. If None, no filtering is applied.
            tokenizer_path: Path to tokenizer for length calculation.
                Required if max_input_len is specified.
            seed: Random seed for shuffle (default: 42).

        Returns:
            Dataset with columns: prompt, gold_answers, silver_answers,
                                  negative_answers, option_keys, identifier
        """
        ds = load_dataset(path, data_dir='PI4PC', split='test')
        original_size = len(ds)

        # Load tokenizer if max_input_len is specified
        tokenizer = None
        if max_input_len is not None and tokenizer_path:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_path, trust_remote_code=True)

        def process_item(item: Dict[str, Any]) -> Dict[str, Any]:
            prompt = _create_pi4pc_prompt(item, prompt_mode)

            # Filter by token length if tokenizer is available
            if tokenizer is not None and max_input_len is not None:
                tokens = tokenizer.encode(prompt, add_special_tokens=False)
                if len(tokens) > max_input_len:
                    return {
                        'prompt': None,
                        'gold_answers': None,
                        'silver_answers': None,
                        'negative_answers': None,
                        'option_keys': None,
                        'identifier': None,
                    }

            # Get option keys for validation
            options = _parse_json_field(item.get('options', {}), {})
            option_keys = list(options.keys()) if isinstance(options, dict) else []

            return {
                'prompt': prompt,
                'gold_answers': item.get('gold_answers', ''),
                'silver_answers': item.get('silver_answers', ''),
                'negative_answers': item.get('negative_answers', ''),
                'option_keys': ','.join(str(k) for k in option_keys),
                'identifier': f"pi4pc_app{item.get('application_number', 'N/A')}_claim{item.get('claim_number', 'N/A')}",
            }

        processed_ds = ds.map(process_item, remove_columns=ds.column_names)

        # Filter out samples that exceeded max_input_len
        if tokenizer is not None and max_input_len is not None:
            processed_ds = processed_ds.filter(lambda x: x['prompt'] is not None)
            filtered_count = original_size - len(processed_ds)
            print(f'[PI4PC] Filtered {filtered_count} samples exceeding '
                  f'{max_input_len} tokens. Remaining: {len(processed_ds)}/{original_size}')

        processed_ds = processed_ds.shuffle(seed=seed)
        return processed_ds


@ICL_EVALUATORS.register_module()
class PI4PCEvaluator(BaseEvaluator):
    """Evaluator for PANORAMA PI4PC task.

    Computes:
    - Gold Hit Rate: Percentage of predictions that hit gold answers
    - Custom Score: (gold*2 + silver*1) / total
    - Accuracy: Overall accuracy
    """

    def score(self, predictions: List[str], references: List[Dict[str, str]],
              test_set: Optional[Dataset] = None) -> Dict[str, Any]:
        """Compute evaluation metrics.

        Args:
            predictions: List of model predictions (raw response strings)
            references: List of dicts with gold_answers, silver_answers, etc.
            test_set: Optional test dataset

        Returns:
            Dictionary containing gold_hit_rate, custom_score, and details
        """
        if len(predictions) != len(references):
            return {'error': 'predictions and references have different length'}

        gold_hits = 0
        silver_hits = 0
        total_scores = 0
        valid_count = 0
        details = []

        for pred_raw, ref in zip(predictions, references):
            # Extract prediction
            pred_key = _extract_answer_from_response(pred_raw)

            # Parse reference answers
            if isinstance(ref, dict):
                gold = _parse_answer_keys(ref.get('gold_answers', ''))
                silver = _parse_answer_keys(ref.get('silver_answers', ''))
                negative = _parse_answer_keys(ref.get('negative_answers', ''))
                option_keys = ref.get('option_keys', '')
            else:
                gold = _parse_answer_keys(ref)
                silver = []
                negative = []
                option_keys = ''

            gold_set = set(gold)
            silver_set = set(silver)

            # Determine category
            is_gold_hit = pred_key is not None and pred_key in gold_set
            is_silver_hit = pred_key is not None and pred_key in silver_set
            is_valid = pred_key is not None

            if is_gold_hit:
                gold_hits += 1
                total_scores += 2
            elif is_silver_hit:
                silver_hits += 1
                total_scores += 1

            if is_valid:
                valid_count += 1

            detail = {
                'pred_raw': pred_raw[:500] if pred_raw else '',
                'pred_extracted': pred_key,
                'gold': gold,
                'silver': silver,
                'is_gold_hit': is_gold_hit,
                'is_silver_hit': is_silver_hit,
                'is_valid': is_valid,
                'correct': is_gold_hit,  # RDAgent compatibility: alias for is_gold_hit
            }
            details.append(detail)

        # Calculate metrics
        total = len(predictions)

        gold_hit_rate = (gold_hits / total * 100) if total > 0 else 0

        # Custom score: average of (gold*2 + silver*1) / count
        # Since we need one prediction per item, max score per item is 2 (if gold hit)
        avg_custom_score = (total_scores / total) if total > 0 else 0
        custom_score_pct = avg_custom_score / 2 * 100  # Normalize to percentage

        # Accuracy: gold or silver hit
        accuracy = ((gold_hits + silver_hits) / total * 100) if total > 0 else 0

        return {
            'gold_hit_rate': gold_hit_rate,
            'custom_score': custom_score_pct,
            'accuracy': accuracy,
            'gold_hits': gold_hits,
            'silver_hits': silver_hits,
            'valid_count': valid_count,
            'total_count': total,
            'details': details,
        }
