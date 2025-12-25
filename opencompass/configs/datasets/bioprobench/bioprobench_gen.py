from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer

# Import dataset and evaluator implementations
from opencompass.datasets.bioprobench.bioprobench_gen import (
	BioProBenchGENDataset,
	BioProBenchGENEvaluator,
	bioprobench_gen_postprocess,
)

QUERY_TEMPLATE = (
	"""
{system_prompt}
{instruction}
Format requirements:
- Each step must be on a separate line.

{input}
""".strip()
)


reader_cfg = dict(
	input_columns=["system_prompt", "instruction", "input"],
	output_column="output",
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


# Evaluation configuration: use custom GEN evaluator
eval_cfg = dict(
	evaluator=dict(type=BioProBenchGENEvaluator),
	pred_postprocessor=dict(type=bioprobench_gen_postprocess),
)


# Dataset registry entry
datasets = [
	dict(
		abbr="BioProBench-GEN",
		type=BioProBenchGENDataset,
		path="bowenxian/BioProBench",
		reader_cfg=reader_cfg,
		infer_cfg=infer_cfg,
		eval_cfg=eval_cfg,
	)
]

