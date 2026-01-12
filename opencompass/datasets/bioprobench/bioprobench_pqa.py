import json
import re
import numpy as np
from sklearn.metrics import brier_score_loss

from datasets import Dataset, load_dataset

from opencompass.registry import LOAD_DATASET, ICL_EVALUATORS

from ..base import BaseDataset
from opencompass.openicl.icl_evaluator import BaseEvaluator

@LOAD_DATASET.register_module()
class BioProBenchPQADataset(BaseDataset):

    @staticmethod
    def load(path="bowenxian/BioProBench", **kwargs):
        ds = load_dataset(path, name="PQA", split="test")
        return ds

def bioprobench_pqa_postprocess(text: str) -> str:
    """
    Extracts the answer and confidence score from a generated string.

    Expected format in the string:
        [ANSWER_START] ... [ANSWER_END] with confidence (as number)

    Returns:
        tuple: (answer: str, confidence: int)

    Raises:
        ValueError: if parsing fails or confidence is invalid.
    """
    # Remove intermediate thinking steps if present
    if '</think>' in text:
        text = text.split("</think>")[-1]

    # Extract content between [ANSWER_START] and [ANSWER_END]
    pattern = r"\[ANSWER_START\](.*?)\[ANSWER_END\]"
    match = re.search(pattern, text, re.DOTALL)
    if not match:
        return None, None
        # raise ValueError("Missing [ANSWER_START] or [ANSWER_END]")

    content = match.group(1).strip()

    # Handle possible answer-confidence formats
    if '&' in content:
        parts = content.split('&')
        if len(parts) != 2:
            return None, None
            # raise ValueError("Expected one '&' to separate answer and confidence")
    else:
        parts = content.split(' ')
        parts = [' '.join(parts[:-1]), parts[-1]]

    answer = parts[0].strip()

    # Extract numerical confidence
    confidence_match = re.search(r"\d+", parts[-1])
    if not confidence_match:
        return None, None
        # raise ValueError("Confidence value not found")
    confidence = int(confidence_match.group())

    if confidence > 100:
        return None, None
        # raise ValueError("Confidence cannot exceed 100")

    return answer, confidence

@ICL_EVALUATORS.register_module()
class BioProBenchPQAEvaluator(BaseEvaluator):
    
    def score(self, predictions: list, references: list) -> dict:
        """
        Compute accuracy and Brier Score given predictions and references.

        predictions: List of tuples (answer: str, confidence: int in [0,100])
        references:  List of correct answers (str)
        """
        total = len(references)
        accs = []
        cfds = []
        failed = 0
        details = []  # RDAgent compatibility: store per-sample details

        for i in range(min(len(predictions), len(references))):
            sample_detail = {
                'answer': None,
                'confidence': None,
                'correct': False,
            }
            try:
                pred = predictions[i]
                ref = references[i]

                # Expect tuple (answer, confidence)
                if pred is None or not isinstance(pred, (list, tuple)) or len(pred) != 2:
                    raise ValueError("Invalid prediction format")

                answer, confidence = pred

                # Basic validation mirroring PQA.py behavior
                if answer is None or confidence is None:
                    raise ValueError("Missing answer or confidence")

                # Ensure confidence is an int and within [0, 100]
                if not isinstance(confidence, (int, float)):
                    # Try to parse numeric value from string
                    m = re.search(r"\d+", str(confidence))
                    if not m:
                        raise ValueError("Confidence value not found")
                    confidence = int(m.group())
                else:
                    confidence = int(confidence)

                if confidence < 0 or confidence > 100:
                    raise ValueError("Confidence out of range")

                is_correct = (answer == ref)
                accs.append(1 if is_correct else 0)
                cfds.append(confidence)

                sample_detail['answer'] = answer
                sample_detail['confidence'] = confidence
                sample_detail['correct'] = is_correct
                details.append(sample_detail)
            except Exception:
                failed += 1
                details.append(sample_detail)

        accuracy = (sum(accs) / len(accs)) if accs else 0.0
        brier = float(brier_score_loss(accs, np.array(cfds) / 100)) if accs else None

        return {
            "accuracy": accuracy * 100,
            "brier_score": brier,  # 均方误差，范围0-1，越低越好，不乘100
            "failed": failed,
            "total": total,
            "details": details,  # RDAgent compatibility: per-sample details
        }

        