import re

from datasets import load_dataset

from opencompass.registry import LOAD_DATASET, ICL_EVALUATORS

from ..base import BaseDataset
from opencompass.openicl.icl_evaluator import BaseEvaluator


@LOAD_DATASET.register_module()
class BioProBenchERRDataset(BaseDataset):
    @staticmethod
    def load(path="bowenxian/BioProBench", seed: int = 42, **kwargs):
        """Load the BioProBench ERR split via HuggingFace datasets.

        Mirrors the PQA/ORD loader style and limits to a small subset
        for quick iteration by default.
        """
        ds = load_dataset(path, name="ERR", split="test")
        # Add derived 'step' column: corrected_text if is_correct else corrupted_text
        def _add_step(example):
            try:
                step = example.get("corrected_text") if example.get("is_correct") else example.get("corrupted_text")
                example["step"] = step
            except Exception:
                # Fallback: keep as None if fields missing
                example["step"] = None
            return example
        ds = ds.map(_add_step)
        ds = ds.shuffle(seed=seed)
        return ds


def bioprobench_err_postprocess(text: str):
    """Extract a binary True/False from model output.

    Rules follow Metrics/ERR.py:
    - Strip trailing thinking and instruction markers.
    - Prefer [ANSWER_START] ... [ANSWER_END] content; otherwise fallback to last line.
    - Accept 'True'/'true' or 'False'/'false'.
    Returns bool on success, or None if parsing fails.
    """
    if text is None:
        return None

    if "</think>" in text:
        text = text.split("</think>")[-1]
    if "[/INST]" in text:
        text = text.split("[/INST]")[-1]

    pattern = r"\[ANSWER_START\](.*?)\[ANSWER_END\]"
    match = re.search(pattern, text, re.DOTALL)

    if match:
        answer = match.group(1).strip()
    else:
        last_line = text.strip().split("\n")[-1].strip()
        answer = last_line

    if "True" in answer or "true" in answer:
        return True
    if "False" in answer or "false" in answer:
        return False

    return None


@ICL_EVALUATORS.register_module()
class BioProBenchERREvaluator(BaseEvaluator):

	def score(self, predictions: list, references: list) -> dict:
		"""Compute accuracy, precision, recall, and F1 for ERR.

		predictions: List of booleans (True/False).
		references:  List of ground-truth booleans or dicts with key 'is_correct'.
		"""
		preds = []
		gts = []
		failed = 0
		total = len(references)
		details = []  # RDAgent compatibility: store per-sample details

		for i in range(min(len(predictions), len(references))):
			sample_detail = {
				'pred': None,
				'gold': None,
				'correct': False,
			}
			try:
				p = predictions[i]
				r = references[i]

				if p is None:
					raise ValueError("Missing prediction")
				if isinstance(r, dict):
					gt = r.get("is_correct")
				else:
					gt = r

				if not isinstance(p, bool) or not isinstance(gt, bool):
					raise ValueError("Non-boolean prediction or reference")

				preds.append(p)
				gts.append(gt)

				is_correct = (p == gt)
				sample_detail['pred'] = p
				sample_detail['gold'] = gt
				sample_detail['correct'] = is_correct
				details.append(sample_detail)
			except Exception:
				failed += 1
				details.append(sample_detail)

		# Classification metrics following Metrics/ERR.py semantics
		if preds:
			TP = sum((p is False and g is False) for p, g in zip(preds, gts))
			FP = sum((p is False and g is True) for p, g in zip(preds, gts))
			FN = sum((p is True and g is False) for p, g in zip(preds, gts))

			accuracy = sum(p == g for p, g in zip(preds, gts)) / len(preds)
			precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
			recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
			f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
		else:
			accuracy = precision = recall = f1 = 0.0

		return {
			"accuracy": accuracy * 100,
			"precision": precision * 100,
			"recall": recall * 100,
			"f1": f1 * 100,
			"failed": failed,
			"total": total,
			"details": details,  # RDAgent compatibility: per-sample details
		}

