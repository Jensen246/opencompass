from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer

# Import dataset and evaluator implementations
from opencompass.datasets.bioprobench.bioprobench_ord import (
	BioProBenchORDDataset,
	BioProBenchORDEvaluator,
	bioprobench_ord_postprocess,
)

QUERY_TEMPLATE = (
	"""
{question}
The steps are:
{wrong_steps}

- Give me the correct order of the steps as a list of their original indices (start from 0), no other words.
- Output your answer *wrapped exactly* between the tags [ANSWER_START] and [ANSWER_END].
- The format of your response must be:
[ANSWER_START]a list of the original indices[ANSWER_END]
""".strip()
)


reader_cfg = dict(
	input_columns=["question", "wrong_steps"],
	output_column="correct_steps",
	test_split="test",
)


# Inference configuration: prompt + retriever + inferencer
infer_cfg = dict(
	prompt_template=dict(
		type=PromptTemplate,
		template=dict(
			round=[
				dict(role="HUMAN", prompt=QUERY_TEMPLATE),
			],
		),
	),
	retriever=dict(type=ZeroRetriever),
	inferencer=dict(type=GenInferencer),
)


# Evaluation configuration: use custom ORD evaluator
eval_cfg = dict(
	evaluator=dict(type=BioProBenchORDEvaluator),
	pred_postprocessor=dict(type=bioprobench_ord_postprocess),
)


# Dataset registry entry
bioprobench_ord_datasets = [
	dict(
		abbr="BioProBench-ORD",
		type=BioProBenchORDDataset,
		path="bowenxian/BioProBench",
		reader_cfg=reader_cfg,
		infer_cfg=infer_cfg,
		eval_cfg=eval_cfg,
	)
]

