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
		ds = load_dataset(path, name="ORD", split="test", **kwargs)

		def _compute(row):
			idx = {s: i for i, s in enumerate(row["wrong_steps"])}
			return {"correct_ids": [idx.get(s) for s in row["correct_steps"]]}

		return ds.map(_compute)


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
		total = len(references)
		preds_indices = []
		gts_indices = []
		failed = 0
		details = []  # RDAgent compatibility: store per-sample details


		for i in range(min(len(predictions), len(references))):
			sample_detail = {
				'pred': None,
				'gold': None,
				'exact_match': False,
				'correct': False,
				'error_msg': None,
			}
			try:
				pred = predictions[i]
				ref = references[i]

				# Normalize prediction into a list of ints
				if pred is None or not isinstance(pred, (list, tuple)):
					raise ValueError("Invalid prediction format")
				try:
					pred = [int(x) for x in pred]
				except Exception:
					raise ValueError("Prediction indices must be integers")

				# Normalize reference into a list of ints
				if isinstance(ref, dict):
					if "correct_ids" in ref and isinstance(ref["correct_ids"], (list, tuple)):
						gt = [int(x) for x in ref["correct_ids"]]
					else:
						raise ValueError("Reference dict missing 'correct_ids' or steps info")
				elif isinstance(ref, (list, tuple)):
					try:
						gt = [int(x) for x in ref]
					except Exception:
						raise ValueError("Reference indices must be integers")
				else:
					raise ValueError("Invalid reference format")

				# Basic validation: same length and same set of items
				if len(pred) != len(gt) or set(pred) != set(gt):
					raise ValueError("Predictions and references must be permutations of the same indices")

				preds_indices.append(pred)
				gts_indices.append(gt)
				is_exact = (pred == gt)
				sample_detail['pred'] = pred
				sample_detail['gold'] = gt
				sample_detail['exact_match'] = is_exact
				sample_detail['correct'] = is_exact
				details.append(sample_detail)
			except Exception as e:
				failed += 1
				sample_detail["error_msg"] = str(e)
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

				shared = gt_rank.keys() & pr_rank.keys()
				for a, b in combinations(shared, 2):
					gt_order = gt_rank[a] - gt_rank[b]
					pr_order = pr_rank[a] - pr_rank[b]
					if gt_order * pr_order > 0:
						concordant_pairs += 1
					total_pairs += 1

			if total_pairs == 0:
				return 0.0
			return (2 * concordant_pairs - total_pairs) / total_pairs

		exact = exact_match(gts_indices, preds_indices)
		tau = kendall_tau(gts_indices, preds_indices)

		return {
			"exact_match": exact * 100,
			"kendall_tau": tau,
			"failed": failed,
			"total": total,
			"details": details,  # RDAgent compatibility: per-sample details
		}

