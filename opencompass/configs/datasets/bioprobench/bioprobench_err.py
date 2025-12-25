from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer

# Import dataset and evaluator implementations
from opencompass.datasets.bioprobench.bioprobench_err import (
	BioProBenchERRDataset,
	BioProBenchERREvaluator,
	bioprobench_err_postprocess,
)

QUERY_TEMPLATE = (
	"""
Determine whether the following target step in a protocol is True or False:
{step}

You may use the following context, which includes the purpose of the step, as well as the preceding and following steps, to inform your decision:
{context}

Please carefully evaluate if the step is logically consistent, necessary, and accurate in the context. If you find anything wrong, answer False.

- Please respond with only True or False, without any additional explanation.
- Output your answer *wrapped exactly* between the tags [ANSWER_START] and [ANSWER_END].
- The format of your response must be:
[ANSWER_START]True or False[ANSWER_END]
""".strip()
)


reader_cfg = dict(
	input_columns=["step", "context"],
	output_column="is_correct",
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


# Evaluation configuration: use custom ERR evaluator
eval_cfg = dict(
	evaluator=dict(type=BioProBenchERREvaluator),
	pred_postprocessor=dict(type=bioprobench_err_postprocess),
)


# Dataset registry entry
datasets = [
	dict(
		abbr="BioProBench-ERR",
		type=BioProBenchERRDataset,
		path="bowenxian/BioProBench",
		reader_cfg=reader_cfg,
		infer_cfg=infer_cfg,
		eval_cfg=eval_cfg,
	)
]

