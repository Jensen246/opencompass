import re
import ast
from itertools import combinations

from datasets import load_dataset

from opencompass.registry import LOAD_DATASET, ICL_EVALUATORS

from ..base import BaseDataset
from opencompass.openicl.icl_evaluator import BaseEvaluator


@LOAD_DATASET.register_module()
class BioProBenchORDDataset(BaseDataset):

	@staticmethod
	def load(path="bowenxian/BioProBench", **kwargs):
		"""Load the BioProBench ORD split via HuggingFace datasets.

		Mirrors the PQA loader style and limits to a small subset
		for quick iteration by default.
		"""
		ds = load_dataset(path, name="ORD", split="test")
		return ds


def bioprobench_ord_postprocess(text: str):
	"""Parse predicted indices list from model output.

	Expects the final answer to be within [ANSWER_START] ... [ANSWER_END]
	and the content to be a Python list of indices, e.g., "[2, 0, 1]".

	Returns a list of ints on success, or None if parsing fails.
	"""
	if "</think>" in text:
		text = text.split("</think>")[-1]

	pattern = r"\[ANSWER_START\](.*?)\[ANSWER_END\]"
	matches = re.findall(pattern, text, re.DOTALL)
	if not matches:
		return None

	index_str = matches[-1].strip()
	try:
		indices = ast.literal_eval(index_str)
	except Exception:
		return None

	if not isinstance(indices, (list, tuple)):
		return None

	try:
		indices = [int(x) for x in indices]
	except Exception:
		return None

	return indices


@ICL_EVALUATORS.register_module()
class BioProBenchORDEvaluator(BaseEvaluator):

	def score(self, predictions: list, references: list) -> dict:
		"""Compute Exact Match and Kendall's Tau for ORD.

		predictions: List of parsed index lists (e.g., [2,0,1]).
		references:  List of samples containing 'wrong_steps' and 'correct_steps'.
		"""
		total = len(references)
		preds_steps = []
		gts_steps = []
		failed = 0
		details = []  # RDAgent compatibility: store per-sample details

		for i in range(min(len(predictions), len(references))):
			sample_detail = {
				'pred': None,
				'gold': None,
				'exact_match': False,
				'correct': False,
			}
			try:
				pred = predictions[i]
				ref = references[i]

				if pred is None or not isinstance(pred, (list, tuple)):
					raise ValueError("Invalid prediction format")

				wrong_steps = None
				correct_steps = None

				if isinstance(ref, dict):
					wrong_steps = ref.get("wrong_steps")
					correct_steps = ref.get("correct_steps")
				else:
					# Fallback for non-dict references if needed
					wrong_steps = getattr(ref, "wrong_steps", None)
					correct_steps = getattr(ref, "correct_steps", None)

				if wrong_steps is None or correct_steps is None:
					raise ValueError("Missing required steps in references")

				if set(pred) != set(range(len(correct_steps))):
					raise ValueError("Invalid or incomplete index set")

				predicted = [wrong_steps[j] for j in pred]
				preds_steps.append(predicted)
				gts_steps.append(correct_steps)

				is_exact = (predicted == correct_steps)
				sample_detail['pred'] = pred
				sample_detail['gold'] = list(range(len(correct_steps)))
				sample_detail['exact_match'] = is_exact
				sample_detail['correct'] = is_exact
				details.append(sample_detail)
			except Exception:
				failed += 1
				details.append(sample_detail)

		def exact_match(gts, preds):
			if not gts:
				return 0.0
			return sum(1 for gt, pr in zip(gts, preds) if gt == pr) / len(gts)

		def kendall_tau(gts, preds):
			total_pairs = 0
			concordant_pairs = 0

			for gt, pr in zip(gts, preds):
				gt_rank = {step: i for i, step in enumerate(gt)}
				pr_rank = {step: i for i, step in enumerate(pr)}

				for a, b in combinations(gt_rank.keys(), 2):
					gt_order = gt_rank[a] - gt_rank[b]
					pr_order = pr_rank[a] - pr_rank[b]
					if gt_order * pr_order > 0:
						concordant_pairs += 1
					total_pairs += 1

			if total_pairs == 0:
				return 0.0
			return (2 * concordant_pairs - total_pairs) / total_pairs

		exact = exact_match(gts_steps, preds_steps)
		tau = kendall_tau(gts_steps, preds_steps)

		return {
			"exact_match": exact * 100,
			"kendall_tau": tau,  # 相关系数，范围-1到1，不乘100
			"failed": failed,
			"total": total,
			"details": details,  # RDAgent compatibility: per-sample details
		}

