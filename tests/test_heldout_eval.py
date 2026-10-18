"""Offline checks of full test answers and TRL's logged evaluation metrics."""
import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"

import tempfile
import unittest
from unittest.mock import patch

from datasets import Dataset
from transformers import AutoTokenizer, LlamaConfig, LlamaForCausalLM
from trl import SFTConfig, SFTTrainer

from src.config import BASE_MODEL, TOKENIZER_REVISION, CHAT_TEMPLATE_KWARGS
from src.test import build_test_dataset, EvaluationMetrics, evaluate_test


class HeldoutEvaluationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tokenizer = AutoTokenizer.from_pretrained(
            BASE_MODEL, revision=TOKENIZER_REVISION, local_files_only=True)
        cls.tokenizer.pad_token = cls.tokenizer.eos_token

    def dataset(self, answers):
        raw = [{"instruction": "Compute 2+2.", "output": answer} for answer in answers]
        manifest = {"references": {"test": [
            {"primary_reference": i} for i in range(len(raw))]}}
        with patch("src.test.MAX_TEST_SAMPLES", len(raw)):
            return build_test_dataset(raw, self.tokenizer, manifest)

    def test_answer_longer_than_training_limit_is_preserved(self):
        data = self.dataset(["reason " * 1100 + "Final answer: 4"])
        ids = self.tokenizer.apply_chat_template(
            data[0]["prompt"] + data[0]["completion"], tokenize=True, return_dict=False,
            **CHAT_TEMPLATE_KWARGS)
        self.assertGreater(len(ids), 1024)
        self.assertTrue(data[0]["completion"][0]["content"].endswith("Final answer: 4"))

    def test_real_trl_evaluation_returns_logged_accuracy(self):
        data = self.dataset(["4", "The answer is 4."])
        model = LlamaForCausalLM(LlamaConfig(
            vocab_size=len(self.tokenizer), hidden_size=8, intermediate_size=16,
            num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2))
        collector = EvaluationMetrics()
        with tempfile.TemporaryDirectory() as tmp:
            args = SFTConfig(
                output_dir=tmp, use_cpu=True, bf16=False, fp16=False,
                per_device_eval_batch_size=1, report_to="none", max_length=None,
                completion_only_loss=True, packing=False, padding_free=False,
                disable_tqdm=True)
            trainer = SFTTrainer(model=model, args=args, train_dataset=data, eval_dataset=data,
                                 processing_class=self.tokenizer, callbacks=[collector])
            metrics = evaluate_test(trainer, collector)
        self.assertIn("test_mean_token_accuracy", metrics)
        self.assertGreater(metrics["test_loss"], 0)
        self.assertGreaterEqual(metrics["test_mean_token_accuracy"], 0)
        self.assertLessEqual(metrics["test_mean_token_accuracy"], 1)


if __name__ == "__main__":
    unittest.main()
