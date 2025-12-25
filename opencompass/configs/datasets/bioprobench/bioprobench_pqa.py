from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer

# Import dataset and evaluator implementations
from opencompass.datasets.bioprobench.bioprobench_pqa import (
	BioProBenchPQADataset,
	BioProBenchPQAEvaluator,
	bioprobench_pqa_postprocess
)

QUERY_TEMPLATE = (
	"""
You are answering a multiple-choice question.
Choose exactly one option from the given Choices.
Return your final answer and confidence in this exact format:
[ANSWER_START] <chosen option text> & <confidence as integer 0-100>% [ANSWER_END]
Only include the final line in that format at the end of your response.

Question:
{question}

Choices:
{choices}
""".strip()
)


reader_cfg = dict(
	input_columns=['question', 'choices'],
	output_column='answer',
	test_split='test',
)


# Inference configuration: prompt + retriever + inferencer
infer_cfg = dict(
	prompt_template=dict(
		type=PromptTemplate,
		template=dict(
			round=[
				dict(role='HUMAN', prompt=QUERY_TEMPLATE),
			],
		),
	),
	retriever=dict(type=ZeroRetriever),
	inferencer=dict(type=GenInferencer),
)


# Evaluation configuration: use custom PQA evaluator
eval_cfg = dict(
	evaluator=dict(type=BioProBenchPQAEvaluator),
    pred_postprocessor=dict(type=bioprobench_pqa_postprocess)
)


# Dataset registry entry
datasets = [
	dict(
		abbr='BioProBench-PQA',
		type=BioProBenchPQADataset,
		path='bowenxian/BioProBench',
		reader_cfg=reader_cfg,
		infer_cfg=infer_cfg,
		eval_cfg=eval_cfg,
	)
]
