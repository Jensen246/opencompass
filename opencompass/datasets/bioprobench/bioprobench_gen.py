import re
import numpy as np
import nltk

from datasets import load_dataset

from opencompass.registry import LOAD_DATASET, ICL_EVALUATORS

from ..base import BaseDataset
from opencompass.openicl.icl_evaluator import BaseEvaluator

from sentence_transformers import SentenceTransformer
from keybert import KeyBERT
from sklearn.metrics.pairwise import cosine_similarity
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score


@LOAD_DATASET.register_module()
class BioProBenchGENDataset(BaseDataset):
    @staticmethod
    def load(path="bowenxian/BioProBench", **kwargs):
        ds = load_dataset(path, name="GEN", split="test")
        return ds


def bioprobench_gen_postprocess(text: str):
    """Extract the final answer content between [ANSWER_START] and [ANSWER_END].

    Strips intermediate thinking or structure blocks if present.
    Returns the cleaned string or None if parsing fails.
    """
    if text is None:
        return None

    if "</think>" in text:
        text = text.split("</think>")[-1]
    if "</Structure>" in text:
        text = text.split("</Structure>")[-1]

    pattern = r"\[ANSWER_START\](.*?)\[ANSWER_END\]"
    match = re.search(pattern, text, re.DOTALL)
    if not match:
        return None
    return match.group(1).strip()


@ICL_EVALUATORS.register_module()
class BioProBenchGENEvaluator(BaseEvaluator):

    SIMILARITY_THRESHOLD = 0.7

    def _ensure_nltk(self):
        try:
            nltk.data.find('tokenizers/punkt')
        except Exception:
            try:
                nltk.download('punkt', quiet=True)
            except Exception:
                pass
        try:
            nltk.data.find('corpora/wordnet')
        except Exception:
            try:
                nltk.download('wordnet', quiet=True)
            except Exception:
                pass

    def _init_models(self):
        # Lazy init models, assuming dependencies are available
        if not hasattr(self, "_embed_initialized"):
            self._embed_initialized = False
            self._embed_available = False
        if not self._embed_initialized:
            self._embed_model = SentenceTransformer('all-mpnet-base-v2')
            self._kw_model = KeyBERT(SentenceTransformer('all-MiniLM-L6-v2'))
            self._embed_available = True
            self._embed_initialized = True

    def _compute_text_metrics(self, reference: str, generated: str):
        bleu = meteor = rouge1 = rouge2 = rougeL = None

        try:
            self._ensure_nltk()
            ref_tokens = nltk.word_tokenize(str(reference).lower())
            gen_tokens = nltk.word_tokenize(str(generated).lower())
            bleu = sentence_bleu([ref_tokens], gen_tokens, weights=(0.5, 0.5),
                                 smoothing_function=SmoothingFunction().method1)
            try:
                meteor = meteor_score([ref_tokens], gen_tokens)
            except Exception:
                meteor = None
        except Exception:
            pass

        try:
            scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
            scores = scorer.score(str(reference), str(generated))
            rouge1 = scores['rouge1'].fmeasure
            rouge2 = scores['rouge2'].fmeasure
            rougeL = scores['rougeL'].fmeasure
        except Exception:
            pass

        return bleu, meteor, rouge1, rouge2, rougeL

    def _compute_keyword_overlap(self, ref_text: str, gen_text: str, top_k: int = 64):
        self._init_models()
        if not self._embed_available:
            return None, None, None
        try:
            ref_kw = set([kw for kw, _ in self._kw_model.extract_keywords(ref_text, top_n=top_k)])
            gen_kw = set([kw for kw, _ in self._kw_model.extract_keywords(gen_text, top_n=top_k)])
        except Exception:
            return None, None, None

        if not ref_kw or not gen_kw:
            return 0.0, 0.0, 0.0

        intersection = ref_kw & gen_kw
        precision = len(intersection) / len(gen_kw)
        recall = len(intersection) / len(ref_kw)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        return precision, recall, f1

    def _compute_step_metrics(self, reference_steps, generated_steps):
        self._init_models()
        if not self._embed_available:
            return None, None
        try:
            ref_embeds = self._embed_model.encode(reference_steps)
            gen_embeds = self._embed_model.encode(generated_steps)
        except Exception:
            return None, None

        matched_refs = set()
        matched_gens = set()

        for i, ref_vec in enumerate(ref_embeds):
            for j, gen_vec in enumerate(gen_embeds):
                try:
                    sim = cosine_similarity([ref_vec], [gen_vec])[0][0]
                except Exception:
                    sim = 0.0
                if sim >= self.SIMILARITY_THRESHOLD:
                    matched_refs.add(i)
                    break

        for i, gen_vec in enumerate(gen_embeds):
            for j, ref_vec in enumerate(ref_embeds):
                try:
                    sim = cosine_similarity([gen_vec], [ref_vec])[0][0]
                except Exception:
                    sim = 0.0
                if sim >= self.SIMILARITY_THRESHOLD:
                    matched_gens.add(i)
                    break

        sr = len(matched_refs) / len(reference_steps) if reference_steps else 1.0
        rp = 1.0 - ((len(generated_steps) - len(matched_gens)) / len(generated_steps)) if generated_steps else 1.0
        return sr, rp

    def score(self, predictions: list, references: list) -> dict:
        bleu_list, meteor_list, rouge1_list, rouge2_list, rougel_list = [], [], [], [], []
        kw_precision_list, kw_recall_list, kw_f1_list = [], [], []
        sr_list, rp_list = [], []
        failed = 0
        total = len(references)

        for i in range(min(len(predictions), len(references))):
            try:
                gen = predictions[i]
                ref = references[i]

                if gen is None:
                    failed += 1
                    continue

                # If reference is a list of steps, compute step metrics
                if isinstance(ref, list):
                    gen_steps = [s.strip() for s in str(gen).split('\n') if s.strip()]
                    sr, rp = self._compute_step_metrics(ref, gen_steps)
                    if sr is not None:
                        sr_list.append(sr)
                    if rp is not None:
                        rp_list.append(rp)
                    ref_text = " ".join(ref)
                else:
                    ref_text = str(ref)

                gen_text = str(gen)

                bleu, meteor, rouge1, rouge2, rougeL = self._compute_text_metrics(ref_text, gen_text)
                if bleu is not None:
                    bleu_list.append(bleu)
                if meteor is not None:
                    meteor_list.append(meteor)
                if rouge1 is not None:
                    rouge1_list.append(rouge1)
                if rouge2 is not None:
                    rouge2_list.append(rouge2)
                if rougeL is not None:
                    rougel_list.append(rougeL)

                kw_p, kw_r, kw_f1 = self._compute_keyword_overlap(ref_text, gen_text)
                if kw_p is not None:
                    kw_precision_list.append(kw_p)
                if kw_r is not None:
                    kw_recall_list.append(kw_r)
                if kw_f1 is not None:
                    kw_f1_list.append(kw_f1)
            except Exception:
                failed += 1

        result = {
            "BLEU": float(np.mean(bleu_list)) * 100 if bleu_list else None,
            "METEOR": float(np.mean(meteor_list)) * 100 if meteor_list else None,
            "ROUGE-1": float(np.mean(rouge1_list)) * 100 if rouge1_list else None,
            "ROUGE-2": float(np.mean(rouge2_list)) * 100 if rouge2_list else None,
            "ROUGE-L": float(np.mean(rougel_list)) * 100 if rougel_list else None,
            "KW_Precision": float(np.mean(kw_precision_list)) * 100 if kw_precision_list else None,
            "KW_Recall": float(np.mean(kw_recall_list)) * 100 if kw_recall_list else None,
            "KW_F1": float(np.mean(kw_f1_list)) * 100 if kw_f1_list else None,
            "Step_Recall": float(np.mean(sr_list)) * 100 if sr_list else None,
            "Redundancy_Penalty": float(np.mean(rp_list)) * 100 if rp_list else None,
            "Failed": failed,
            "Total": total,
        }

        return result

