from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Literal, Optional, Tuple

import numpy as np
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

from experiments.audit.unlearning_audit_reporter import parse_judge_json
from llm_utils.gemini_client import GeminiClient

logger = logging.getLogger(__name__)

ClassificationFailure = Literal["api", "parse"]
TEXT_SAMPLE_PLACEHOLDER = "{{TEXT_SAMPLE}}"


class UnlearningTargetEvaluator:
    """
    Domain manager for blind classification validation experiments.

    Uses a (typically faster) Gemini client instance independent of the audit
    reporter's model configuration to compute Balanced Accuracy and AUC-ROC.
    """

    def __init__(
        self,
        client: GeminiClient,
        *,
        seed: int = 42,
    ) -> None:
        self.client = client
        self.rng = np.random.default_rng(seed)

    @staticmethod
    def _build_classification_prompt_template(hypothesis: str) -> str:
        """Few-shot prompt template, substitute ``TEXT_SAMPLE_PLACEHOLDER`` per sample."""
        return (
            "You are an expert academic evaluator verifying a machine unlearning process.\n"
            "Your task is to judge whether a given text sample belongs to a specific "
            "target forgotten concept.\n\n"
            "# ====== FEW-SHOT EXAMPLES ====== #\n\n"
            "Example 1: Clear Concept Match\n"
            "Target Forgotten Concept: \"Differential Calculus and Derivatives\"\n"
            "Text Sample: \"The instantaneous rate of change is found by taking the "
            "limit as delta x approaches zero, yielding the derivative function.\"\n"
            "Response:\n"
            "{\n"
            '  "belongs_to_forget_concept": true,\n'
            '  "forget_probability": 0.95\n'
            "}\n\n"
            "Example 2: Complete Concept Mismatch (Retained Concept)\n"
            "Target Forgotten Concept: \"Differential Calculus and Derivatives\"\n"
            "Text Sample: \"Photosynthesis in green plants converts solar energy into "
            "chemical energy by processing carbon dioxide and water into glucose.\"\n"
            "Response:\n"
            "{\n"
            '  "belongs_to_forget_concept": false,\n'
            '  "forget_probability": 0.01\n'
            "}\n\n"
            "Example 3: Marginal/Ambiguous Contextual Match\n"
            "Target Forgotten Concept: \"Differential Calculus and Derivatives\"\n"
            "Text Sample: \"We plotted the continuous functions on a 2D grid to track "
            "how the graph slopes upwards during the simulation run.\"\n"
            "Response:\n"
            "{\n"
            '  "belongs_to_forget_concept": true,\n'
            '  "forget_probability": 0.65\n'
            "}\n\n"
            "# ====== LIVE EVALUATION ====== #\n\n"
            f"Target Forgotten Concept: {json.dumps(hypothesis)}\n\n"
            "Text Sample to Evaluate:\n"
            f"\"\"\"{TEXT_SAMPLE_PLACEHOLDER}\"\"\"\n\n"
            "Instructions:\n"
            "1. Evaluate if the live text sample directly relates to, discusses, or "
            "exemplifies the Target Forgotten Concept.\n"
            "2. Assign a strict continuous probability score representing the "
            "likelihood that the text belongs to the forgotten concept (Class 1).\n"
            "   - A score near 1.0 means absolute certainty that the text is related "
            "to the forgotten concept.\n"
            "   - A score near 0.0 means absolute certainty that the text is completely "
            "unrelated (belongs to the retained set).\n"
            "3. Do not include any thoughts, reasoning, markdown formatting, code "
            "blocks, or conversational text. Your output must be a single parseable "
            "JSON object matching the schema shown in the examples above and nothing "
            "else.\n\n"
            "JSON Response Schema:\n"
            "{\n"
            '  "belongs_to_forget_concept": true,\n'
            '  "forget_probability": 0.5\n'
            "}"
        )

    @classmethod
    def _build_classification_prompt(cls, hypothesis: str, text_sample: str) -> str:
        """Construct a strict few-shot binary classification prompt for the LLM."""
        return cls._build_classification_prompt_template(hypothesis).replace(
            TEXT_SAMPLE_PLACEHOLDER,
            text_sample,
        )

    def classify_sample(
        self,
        hypothesis: str,
        text_sample: str,
    ) -> Tuple[Optional[bool], Optional[float], Optional[ClassificationFailure]]:
        """
        Few-shot inference for a single validation text.

        Returns:
            ``(is_forget, forget_probability, failure)``. On success ``failure`` is
            ``None``; on API or parse errors both predictions are ``None``.
        """
        prompt = self._build_classification_prompt(hypothesis, text_sample)
        return self._classify_with_prompt(prompt)

    def _classify_with_prompt(
        self,
        prompt: str,
    ) -> Tuple[Optional[bool], Optional[float], Optional[ClassificationFailure]]:
        """Run classification on a pre-built prompt."""
        raw_text, error, _ = self.client.generate_text(prompt)

        if error or not raw_text:
            logger.warning(
                "Classification API failed (%s); sample excluded from metrics.",
                error or "empty response",
            )
            return None, None, "api"

        try:
            data = parse_judge_json(raw_text)
            if data.get("_parse_error"):
                raise ValueError("unparseable classification response")

            raw_is_forget = data.get("belongs_to_forget_concept")
            if raw_is_forget is True:
                is_forget = True
            elif raw_is_forget is False:
                is_forget = False
            else:
                raise ValueError(
                    f"belongs_to_forget_concept must be a strict boolean, got: {raw_is_forget!r}"
                )
            if "forget_probability" not in data:
                raise ValueError(
                    "forget_probability is missing from classification response"
                )
            prob = float(data["forget_probability"])
            if not np.isfinite(prob) or not (0.0 <= prob <= 1.0):
                raise ValueError(
                    f"forget_probability must be a finite value in [0, 1], got {prob!r}"
                )

            # Align probability direction: high score should mean class 1 (forget).
            if not is_forget and prob > 0.5:
                prob = 1.0 - prob
            elif is_forget and prob < 0.5:
                prob = max(prob, 1.0 - prob)

            return is_forget, prob, None
        except (TypeError, ValueError, KeyError) as exc:
            logger.error(
                "Failed to parse classification JSON (%s); sample excluded from metrics.",
                exc,
            )
            return None, None, "parse"

    def prepare_blind_evaluation_set(
        self,
        forget_samples: List[str],
        retain_samples: List[str],
        max_samples_per_set: int = 25,
    ) -> Tuple[List[str], List[int]]:
        """
        Sample, balance, and shuffle forget and retain examples.

        Returns:
            ``(shuffled_texts, ground_truth_labels)`` where label ``1`` = forget,
            ``0`` = retain.
        """
        if not forget_samples or not retain_samples:
            raise ValueError(
                "Both forget_samples and retain_samples must be non-empty; got "
                f"{len(forget_samples)} forget and {len(retain_samples)} retain."
            )

        n = min(len(forget_samples), len(retain_samples), max_samples_per_set)
        if n == 0:
            raise ValueError(
                "Need at least one sample per class after applying max_samples_per_set, "
                f"got forget_pool={len(forget_samples)}, retain_pool={len(retain_samples)}, "
                f"max_samples_per_set={max_samples_per_set}."
            )

        sampled_forget = self.rng.choice(
            forget_samples, size=n, replace=False,
        ).tolist()
        sampled_retain = self.rng.choice(
            retain_samples, size=n, replace=False,
        ).tolist()

        combined_texts = sampled_forget + sampled_retain
        labels = [1] * n + [0] * n

        indices = np.arange(len(combined_texts))
        self.rng.shuffle(indices)

        shuffled_texts = [combined_texts[int(i)] for i in indices]
        ground_truth_labels = [labels[int(i)] for i in indices]
        return shuffled_texts, ground_truth_labels

    @staticmethod
    def _compute_metrics(
        y_true: np.ndarray,
        y_pred_binary: np.ndarray,
        y_pred_prob: np.ndarray,
    ) -> Tuple[Optional[float], Optional[float]]:
        """Return ``(balanced_accuracy, auc_roc)``, or ``None`` when undefined."""
        if len(y_true) == 0:
            return None, None

        balanced_acc = float(balanced_accuracy_score(y_true, y_pred_binary))

        if len(np.unique(y_true)) < 2:
            logger.warning(
                "AUC-ROC undefined: ground truth contains a single class. "
                "Returning None for auc_roc."
            )
            return balanced_acc, None

        try:
            auc_roc = float(roc_auc_score(y_true, y_pred_prob))
        except ValueError:
            logger.warning(
                "AUC-ROC undefined (e.g. constant predicted scores). "
                "Returning None for auc_roc."
            )
            auc_roc = None

        return balanced_acc, auc_roc

    def evaluate_hypothesis(
        self,
        hypothesis: str,
        forget_samples: List[str],
        retain_samples: List[str],
        max_samples_per_set: int = 25,
    ) -> Dict[str, Any]:
        """
        Run blind classification over a mixed corpus and compute metrics.

        Metrics are computed only on successfully classified samples, API and
        parse failures are counted but excluded so fallbacks do not
        inflate scores.

        Returns:
            Report with balanced accuracy, AUC-ROC (when defined), predictions,
            and failure counts.
        """
        test_texts, y_true = self.prepare_blind_evaluation_set(
            forget_samples=forget_samples,
            retain_samples=retain_samples,
            max_samples_per_set=max_samples_per_set,
        )
        prompt_template = self._build_classification_prompt_template(hypothesis)

        predicted_labels: List[Optional[int]] = []
        predicted_probabilities: List[Optional[float]] = []
        sample_records: List[Dict[str, Any]] = []
        n_api_failures = 0
        n_parse_failures = 0

        y_true_scored: List[int] = []
        y_pred_binary_scored: List[int] = []
        y_pred_prob_scored: List[float] = []

        logger.info(
            "Starting blind hypothesis evaluation for target %r over %d text samples.",
            hypothesis,
            len(test_texts),
        )

        for index, (text, label) in enumerate(zip(test_texts, y_true)):
            prompt = self._build_classification_prompt(hypothesis, text)
            is_forget, prob, failure = self._classify_with_prompt(prompt)
            if failure == "api":
                n_api_failures += 1
            elif failure == "parse":
                n_parse_failures += 1

            if failure is not None:
                predicted_labels.append(None)
                predicted_probabilities.append(None)
                sample_records.append(
                    {
                        "index": index,
                        "text": text,
                        "ground_truth": label,
                        "ground_truth_class": "forget" if label == 1 else "retain",
                        "predicted_label": None,
                        "forget_probability": None,
                        "failure": failure,
                    }
                )
                continue

            assert is_forget is not None and prob is not None
            predicted_labels.append(int(is_forget))
            predicted_probabilities.append(prob)
            y_true_scored.append(label)
            y_pred_binary_scored.append(int(is_forget))
            y_pred_prob_scored.append(prob)
            sample_records.append(
                {
                    "index": index,
                    "text": text,
                    "ground_truth": label,
                    "ground_truth_class": "forget" if label == 1 else "retain",
                    "predicted_label": int(is_forget),
                    "forget_probability": prob,
                    "failure": None,
                }
            )

        y_true_np = np.array(y_true, dtype=int)
        y_true_scored_np = np.array(y_true_scored, dtype=int)
        y_pred_bin_scored_np = np.array(y_pred_binary_scored, dtype=int)
        y_pred_prob_scored_np = np.array(y_pred_prob_scored, dtype=float)

        balanced_acc, auc_roc = self._compute_metrics(
            y_true_scored_np, y_pred_bin_scored_np, y_pred_prob_scored_np,
        )

        if n_api_failures or n_parse_failures:
            logger.warning(
                "Evaluation completed with %d API and %d parse failures "
                "(%d/%d samples scored).",
                n_api_failures,
                n_parse_failures,
                len(y_true_scored),
                len(test_texts),
            )

        return {
            "hypothesis": hypothesis,
            "_classification_prompt_template": prompt_template,
            "balanced_accuracy": balanced_acc,
            "auc_roc": auc_roc,
            "n_samples_evaluated": len(test_texts),
            "n_samples_scored": len(y_true_scored),
            "n_api_failures": n_api_failures,
            "n_parse_failures": n_parse_failures,
            "ground_truth": y_true_np.tolist(),
            "predicted_labels": predicted_labels,
            "predicted_probabilities": predicted_probabilities,
            "samples": sample_records,
        }
