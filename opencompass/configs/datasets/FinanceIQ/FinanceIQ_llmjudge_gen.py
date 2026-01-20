from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import FixKRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets import FinanceIQDataset
from opencompass.evaluator import GenericLLMEvaluator
from opencompass.datasets import generic_llmjudge_postprocess

financeIQ_subject_mapping = {
    '注册会计师（CPA）': '注册会计师（CPA）',
    '银行从业资格': '银行从业资格',
    '证券从业资格': '证券从业资格',
    '基金从业资格': '基金从业资格',
    '保险从业资格CICE': '保险从业资格CICE',
    '经济师': '经济师',
    '税务师': '税务师',
    '期货从业资格': '期货从业资格',
    '理财规划师': '理财规划师',
    '精算师-金融数学': '精算师-金融数学',
}

QUERY_TEMPLATE = '''以下是关于{_ch_name}的单项选择题，请给出正确答案。

题目：{{question}}
A. {{A}}
B. {{B}}
C. {{C}}
D. {{D}}
'''

GRADER_TEMPLATE = '''请作为评分专家，判断候选答案是否与标准答案一致。

评分标准：
1. 只需判断候选答案选择的选项是否与标准答案相同
2. 忽略表达方式的差异，只看最终选择的选项
3. 如果候选答案中明确选择了某个选项（A/B/C/D），以该选项为准

请根据以下信息判断：

<标准答案>{answer}</标准答案>

<候选答案>{prediction}</候选答案>

判断结果（只返回字母）：
A - 答案正确
B - 答案错误
'''

financeIQ_all_sets = list(financeIQ_subject_mapping.keys())

financeIQ_datasets = []
for _name in financeIQ_all_sets:
    _ch_name = financeIQ_subject_mapping[_name]
    financeIQ_infer_cfg = dict(
        ice_template=dict(
            type=PromptTemplate,
            template=dict(
                begin='</E>',
                round=[
                    dict(
                        role='HUMAN',
                        prompt=QUERY_TEMPLATE.format(_ch_name=_ch_name)
                    ),
                    dict(role='BOT', prompt='答案是: {answer}'),
                ]),
            ice_token='</E>',
        ),
        retriever=dict(type=FixKRetriever, fix_id_list=[0, 1, 2, 3, 4]),
        inferencer=dict(type=GenInferencer),
    )

    financeIQ_eval_cfg = dict(
        evaluator=dict(
            type=GenericLLMEvaluator,
            prompt_template=dict(
                type=PromptTemplate,
                template=dict(
                    begin=[
                        dict(
                            role='SYSTEM',
                            fallback_role='HUMAN',
                            prompt='You are a grading expert for Chinese financial certification exams.',
                        )
                    ],
                    round=[
                        dict(role='HUMAN', prompt=GRADER_TEMPLATE),
                    ],
                ),
            ),
            dataset_cfg=dict(
                type=FinanceIQDataset,
                path='./data/FinanceIQ/',
                name=_name,
                reader_cfg=dict(
                    input_columns=['question', 'A', 'B', 'C', 'D'],
                    output_column='answer',
                    train_split='dev',
                    test_split='test',
                ),
            ),
            judge_cfg=dict(),
            dict_postprocessor=dict(type=generic_llmjudge_postprocess),
        ),
        pred_role='BOT',
    )

    financeIQ_datasets.append(
        dict(
            type=FinanceIQDataset,
            path='./data/FinanceIQ/',
            name=_name,
            abbr=f'FinanceIQ-{_name}',
            reader_cfg=dict(
                input_columns=['question', 'A', 'B', 'C', 'D'],
                output_column='answer',
                train_split='dev',
                test_split='test'),
            infer_cfg=financeIQ_infer_cfg,
            eval_cfg=financeIQ_eval_cfg,
        ))

del _name, _ch_name
