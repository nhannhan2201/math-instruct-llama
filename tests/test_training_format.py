"""CPU offline protocol regressions; no pretrained weights or model forward."""
import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path
import json
from datasets import Dataset
from transformers import AutoTokenizer, LlamaConfig, LlamaForCausalLM
from trl import SFTConfig, SFTTrainer
from src.config import BASE_MODEL, TOKENIZER_REVISION, CHAT_TEMPLATE_KWARGS
from src.data_prep import build_prompt, format_batch, prepared_from_manifest, tokenizer_provenance

class TrainingFormatTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, revision=TOKENIZER_REVISION, local_files_only=True)

    def test_actual_ids_labels_padding_and_parity(self):
        t = self.tokenizer
        rows = {"instruction": [" What is 2 + 2? ", "Compute 3 times 7."],
                "output": [" 4 ", "def solution():\n    return 3 * 7"]}
        data = Dataset.from_dict(format_batch(rows, t))
        model = LlamaForCausalLM(LlamaConfig(vocab_size=len(t), hidden_size=8,
            intermediate_size=16, num_hidden_layers=1, num_attention_heads=2,
            num_key_value_heads=2, bos_token_id=t.bos_token_id,
            eos_token_id=t.eos_token_id, pad_token_id=t.pad_token_id))
        with tempfile.TemporaryDirectory() as tmp:
            args = SFTConfig(output_dir=tmp, use_cpu=True, bf16=False, fp16=False,
                report_to="none", optim="adamw_torch", gradient_checkpointing=False,
                max_length=1024, packing=False, padding_free=False,
                completion_only_loss=True, assistant_only_loss=False, disable_tqdm=True)
            trainer = SFTTrainer(model=model, args=args, train_dataset=data, processing_class=t)
            features = list(trainer.train_dataset)
            batch = trainer.data_collator(features)
            for i, example in enumerate(data):
                prefix = t.apply_chat_template(example["prompt"], tokenize=True, return_dict=False,
                    add_generation_prompt=True, **CHAT_TEMPLATE_KWARGS)
                ids = t.apply_chat_template(example["prompt"] + example["completion"],
                    tokenize=True, return_dict=False, **CHAT_TEMPLATE_KWARGS)
                self.assertEqual(features[i]["input_ids"], ids)
                self.assertEqual(ids[:len(prefix)], prefix)
                self.assertEqual(ids[0], t.bos_token_id)
                self.assertEqual(ids.count(t.bos_token_id), 1)
                self.assertEqual(ids[-1], t.convert_tokens_to_ids("<|eot_id|>"))
                self.assertEqual(ids.count(t.convert_tokens_to_ids("<|eot_id|>")), 3)
                header = "<|start_header_id|>assistant<|end_header_id|>\n\n"
                rendered = t.apply_chat_template(build_prompt(rows["instruction"][i]),
                    tokenize=False, add_generation_prompt=True, **CHAT_TEMPLATE_KWARGS)
                self.assertTrue(rendered.endswith(header))
                self.assertEqual(t(rendered, add_special_tokens=False)["input_ids"], prefix)
                self.assertEqual(features[i]["completion_mask"], [0]*len(prefix)+[1]*(len(ids)-len(prefix)))
                width = batch["input_ids"].shape[1]
                self.assertEqual(batch["input_ids"][i][:len(ids)].tolist(), ids)
                self.assertEqual(batch["attention_mask"][i].tolist(), [1]*len(ids)+[0]*(width-len(ids)))
                self.assertEqual(batch["labels"][i].tolist(), [-100]*len(prefix)+ids[len(prefix):]+[-100]*(width-len(ids)))
                self.assertGreater(sum(x != -100 for x in batch["labels"][i][1:len(ids)-1]), 0)
            self.assertTrue((batch["attention_mask"] == 0).any())
        from src.inference import MathSolver
        solver = MathSolver.__new__(MathSolver)
        solver.tokenizer = t
        calls = []
        solver.generator = lambda prompt, **kwargs: calls.append((prompt, kwargs)) or [{"generated_text": "4"}]
        self.assertEqual(solver.solve(rows["instruction"][0]), "4")
        self.assertEqual(t(calls[0][0], add_special_tokens=False)["input_ids"],
                         t.apply_chat_template(data[0]["prompt"], tokenize=True, return_dict=False, add_generation_prompt=True, **CHAT_TEMPLATE_KWARGS))
        self.assertFalse(calls[0][1]["add_special_tokens"])
        self.assertEqual(calls[0][1]["eos_token_id"], t.convert_tokens_to_ids("<|eot_id|>"))

    def test_overlength_and_invalid(self):
        t = self.tokenizer
        raw = Dataset.from_list([{"instruction": "2+2?", "output": "4"},
                                 {"instruction": "Explain.", "output": "reason "*200}])
        m = {"pairs": {"a": {"raw_indices": [0]}, "b": {"raw_indices": [1]}},
             "training_subset": ["a", "b"], "references": {"validation": [{"group_id": "v", "primary_reference": 0}]},
             "revision": "fixture", "snapshot_hash": "fixture"}
        example = {k: v[0] for k,v in format_batch({"instruction": ["2+2?"], "output": ["4"]}, t).items()}
        size = len(t.apply_chat_template(example["prompt"]+example["completion"], tokenize=True, return_dict=False, **CHAT_TEMPLATE_KWARGS))
        train, val, meta = prepared_from_manifest(raw,t,m,return_metadata=True,max_length=size)
        self.assertEqual([x["pair_id"] for x in meta["effective"]], ["a"])
        self.assertEqual([x["pair_id"] for x in meta["excluded"]], ["b"])
        self.assertEqual(len(meta["selected"]), 2)
        with self.assertRaisesRegex(ValueError, "Validation exceeds"):
            prepared_from_manifest(raw,t,m,max_length=size-1)
        for answer in ["", "   "]:
            with self.assertRaisesRegex(ValueError, "answer"):
                format_batch({"instruction": ["Q"], "output": [answer]}, t)
        with self.assertRaisesRegex(ValueError, "question"):
            build_prompt(" ")

    def test_inference_tokenizer_compatibility(self):
        from src.inference import MathSolver
        t = self.tokenizer
        with tempfile.TemporaryDirectory() as tmp:
            adapter = Path(tmp, "qlora_final"); adapter.mkdir()
            with patch("src.inference.OUTPUT_ROOT", tmp), patch("src.inference.AutoModelForCausalLM.from_pretrained", side_effect=RuntimeError("weights blocked")) as weights:
                with self.assertRaisesRegex(ValueError, "provenance"):
                    MathSolver()
                weights.assert_not_called()
                metadata = {"tokenizer": tokenizer_provenance(t)}
                Path(adapter,"effective_data.json").write_text(json.dumps(metadata))
                with patch("src.inference.AutoTokenizer.from_pretrained", return_value=t) as loader:
                    with self.assertRaisesRegex(RuntimeError,"weights blocked"):
                        MathSolver()
                    self.assertEqual(loader.call_args.args[0], BASE_MODEL)
                    self.assertTrue(loader.call_args.kwargs["local_files_only"])
                from src.train import save_protocol
                save_protocol(adapter, t, metadata)
                loaded = AutoTokenizer.from_pretrained(adapter,local_files_only=True)
                self.assertEqual(tokenizer_provenance(loaded), metadata["tokenizer"])
                metadata["tokenizer"]["chat_template_kwargs"] = {"date_string": "wrong"}
                Path(adapter,"effective_data.json").write_text(json.dumps(metadata))
                weights.reset_mock()
                with self.assertRaisesRegex(ValueError, "differs"):
                    MathSolver()
                weights.assert_not_called()

class TrainingRuntimeContractTests(unittest.TestCase):
    """Production wiring with mocks and tiny randomly initialized adapters only."""

    def test_revision_loader_and_negative_inputs(self):
        from types import SimpleNamespace
        from src.config import DATASET_ID, DATASET_REVISION
        from src.data_prep import cached_revision, prepare_data
        raw = SimpleNamespace(info=SimpleNamespace(download_checksums={
            f"hf://datasets/{DATASET_ID}@{DATASET_REVISION}/MathInstruct.json": {}}))
        self.assertEqual(cached_revision(raw), DATASET_REVISION)
        raw.info.download_checksums = {f"hf://datasets/{DATASET_ID}@{'0'*40}/file": {}}
        with self.assertRaisesRegex(ValueError, "pinned revision"):
            cached_revision(raw)
        raw.info.download_checksums = {}
        with self.assertRaisesRegex(ValueError, "source metadata"):
            cached_revision(raw)
        with patch("src.data_prep.load_dataset", side_effect=FileNotFoundError("offline cache missing")) as loader:
            with self.assertRaises(FileNotFoundError):
                prepare_data(None)
            loader.assert_called_once_with(DATASET_ID, revision=DATASET_REVISION)
        with self.assertRaisesRegex(ValueError, "Unequal"):
            format_batch({"instruction": ["Q"], "output": []}, None)
        for invalid in (0, -1, True, 1.5, "1024"):
            with self.assertRaisesRegex(ValueError, "positive integer"):
                prepared_from_manifest(None, None, {}, max_length=invalid)

    def test_config_and_smoke_selection(self):
        from src.train import training_config, smoke_subset, CHECKPOINTING_KWARGS
        from src.data_prep import digest
        with tempfile.TemporaryDirectory() as tmp:
            smoke = training_config(tmp, smoke=True, bf16=False)
            full = training_config(tmp, smoke=False, bf16=False)
        self.assertEqual(smoke.max_steps, 2)
        self.assertEqual(full.max_steps, -1)
        self.assertEqual(smoke.gradient_accumulation_steps, 32)
        self.assertTrue(smoke.fp16)
        self.assertFalse(smoke.bf16)
        self.assertTrue(smoke.gradient_checkpointing)
        self.assertEqual(smoke.gradient_checkpointing_kwargs, CHECKPOINTING_KWARGS)
        self.assertTrue(smoke.completion_only_loss)
        self.assertFalse(smoke.packing or smoke.padding_free or smoke.assistant_only_loss)
        for field in ("learning_rate", "warmup_steps", "num_train_epochs", "optim", "seed"):
            self.assertEqual(getattr(smoke, field), getattr(full, field))
        ds = Dataset.from_dict({"id": list(range(80))})
        meta = {"effective": [{"pair_id": str(i)} for i in range(80)]}
        self.assertEqual(list(smoke_subset(ds, meta)["id"]), list(range(64)))
        self.assertEqual(meta["training_selection"]["ids_hash"], digest(meta["effective"][:64]))
        with self.assertRaisesRegex(ValueError, "64"):
            smoke_subset(ds.select(range(63)), meta)

    def test_seed_initialization_and_main_wiring(self):
        from contextlib import ExitStack
        from types import SimpleNamespace
        from unittest.mock import MagicMock
        import torch
        from src import train
        from src.config import BASE_REVISION, TOKENIZER_REVISION
        events = []
        parameter = torch.nn.Parameter(torch.zeros(1))
        model = MagicMock()
        model.named_parameters.return_value = [("lora_A", parameter)]
        trainer = MagicMock()
        trainer.state.global_step = 2
        trainer.evaluate.return_value = {"eval_loss": 1.0}
        def train_two():
            events.append("train")
            with torch.no_grad():
                parameter.add_(1)
        trainer.train.side_effect = train_two
        ds = Dataset.from_dict({"id": list(range(80))})
        metadata = {"effective": [{"pair_id": str(i)} for i in range(80)],
                    "effective_ids_hash": "fixture", "revision": "fixture"}
        run = MagicMock()
        run.__enter__.return_value = SimpleNamespace(info=SimpleNamespace(run_id="fixture"))
        with tempfile.TemporaryDirectory() as tmp, ExitStack() as stack:
            stack.enter_context(patch.object(train, "OUTPUT_ROOT", tmp))
            stack.enter_context(patch.object(train.torch.cuda, "is_available", return_value=True))
            stack.enter_context(patch.object(train.torch.cuda, "device_count", return_value=1))
            stack.enter_context(patch.object(train.torch.cuda, "is_bf16_supported", side_effect=lambda including_emulation=True: including_emulation))
            for name in ("set_device", "synchronize", "reset_peak_memory_stats", "get_device_name", "max_memory_allocated", "max_memory_reserved"):
                stack.enter_context(patch.object(train.torch.cuda, name, return_value=0))
            for name in ("set_tracking_uri", "set_experiment", "log_params", "log_metric", "log_metrics", "log_dict", "log_artifacts"):
                stack.enter_context(patch.object(train.mlflow, name))
            stack.enter_context(patch.object(train.mlflow, "start_run", return_value=run))
            stack.enter_context(patch.object(train, "set_seed", side_effect=lambda seed: events.append("seed")))
            token_loader = stack.enter_context(patch.object(train.AutoTokenizer, "from_pretrained", side_effect=lambda *a, **kw: events.append("tokenizer") or MagicMock()))
            base_loader = stack.enter_context(patch.object(train.AutoModelForCausalLM, "from_pretrained", side_effect=lambda *a, **kw: events.append("base") or model))
            stack.enter_context(patch.object(train, "prepare_data", return_value=(ds, ds.select(range(1)), metadata)))
            stack.enter_context(patch.object(train, "get_peft_model", side_effect=lambda *a: events.append("adapter") or model))
            trainer_class = stack.enter_context(patch.object(train, "SFTTrainer", return_value=trainer))
            protocol = stack.enter_context(patch.object(train, "save_protocol"))
            train.main(["--method", "lora", "--smoke"])
            self.assertEqual(events, ["seed", "tokenizer", "base", "adapter", "train"])
            self.assertEqual(base_loader.call_args.kwargs["revision"], BASE_REVISION)
            self.assertEqual(base_loader.call_args.kwargs["dtype"], torch.float16)
            self.assertTrue(trainer_class.call_args.kwargs["args"].fp16)
            self.assertFalse(trainer_class.call_args.kwargs["args"].bf16)
            self.assertEqual(token_loader.call_args.kwargs["revision"], TOKENIZER_REVISION)
            self.assertEqual(trainer_class.call_args.kwargs["args"].max_steps, 2)
            self.assertEqual(len(trainer_class.call_args.kwargs["train_dataset"]), 64)
            trainer.evaluate.assert_called_once()
            trainer.save_model.assert_called_once()
            self.assertIn("lora_smoke_", str(protocol.call_args.args[0]))
            self.assertEqual(protocol.call_args.args[0].name, "final")
            events.clear()
            model.is_loaded_in_4bit = True
            with patch.object(train, "prepare_model_for_kbit_training", side_effect=lambda m, **kw: events.append("kbit") or m) as kbit:
                train.main(["--method", "qlora", "--smoke"])
                self.assertEqual(events, ["seed", "tokenizer", "base", "kbit", "adapter", "train"])
                self.assertEqual(kbit.call_args.kwargs["gradient_checkpointing_kwargs"], train.CHECKPOINTING_KWARGS)
                quantization = base_loader.call_args.kwargs["quantization_config"]
                self.assertTrue(quantization.load_in_4bit)
                self.assertEqual(quantization.bnb_4bit_quant_type, "nf4")
                self.assertTrue(quantization.bnb_4bit_use_double_quant)
                self.assertEqual(trainer.evaluate.call_count, 2)


    def test_random_adapter_and_mlflow_round_trip(self):
        import torch
        import mlflow
        from mlflow.tracking import MlflowClient
        from peft import LoraConfig, get_peft_model, PeftModel
        from src.config import BASE_REVISION, BASE_MODEL
        from src.train import save_protocol
        torch.manual_seed(42)
        cfg = LlamaConfig(vocab_size=32, hidden_size=8, intermediate_size=16,
                          num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2)
        base = LlamaForCausalLM(cfg)
        base.config._name_or_path = BASE_MODEL
        model = get_peft_model(base, LoraConfig(r=4, lora_alpha=8, target_modules=["q_proj"],
                                               task_type="CAUSAL_LM", revision=BASE_REVISION))
        expected = {n: p.detach().clone() for n,p in model.named_parameters() if p.requires_grad}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp, "final")
            model.save_pretrained(path)
            # Stub tokenizer isolates artifact contract; actual tokenizer round-trip tested above.
            from unittest.mock import MagicMock
            save_protocol(path, MagicMock(), {"base_revision": BASE_REVISION})
            reloaded = PeftModel.from_pretrained(LlamaForCausalLM(cfg), path, is_trainable=True)
            for n,p in reloaded.named_parameters():
                if n in expected:
                    self.assertTrue(torch.equal(p, expected[n]))
            self.assertEqual(json.loads((path / "adapter_config.json").read_text())["revision"], BASE_REVISION)
            old_uri = mlflow.get_tracking_uri()
            try:
                mlflow.set_tracking_uri(f"sqlite:///{tmp}/tracking.db")
                mlflow.create_experiment("offline-contract", artifact_location=str(Path(tmp, "artifacts")))
                mlflow.set_experiment("offline-contract")
                with mlflow.start_run() as run:
                    mlflow.log_metric("final_eval_loss", 1.25, step=2)
                    mlflow.log_artifacts(str(path), "final")
                    run_id = run.info.run_id
                client = MlflowClient()
                self.assertEqual(client.get_run(run_id).data.metrics["final_eval_loss"], 1.25)
                downloaded = Path(client.download_artifacts(run_id, "final/effective_data.json", tmp))
                self.assertEqual(json.loads(downloaded.read_text()), {"base_revision": BASE_REVISION})
            finally:
                mlflow.set_tracking_uri(old_uri)

    def test_two_optimizer_steps_with_random_cpu_model(self):
        from dataclasses import replace
        import torch
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel
        from transformers import PreTrainedTokenizerFast, set_seed
        from peft import LoraConfig, get_peft_model
        from src.train import training_config
        tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=Tokenizer(WordLevel({"<pad>": 0, "<unk>": 1, "Q": 2, "A": 3}, unk_token="<unk>")),
            pad_token="<pad>", unk_token="<unk>")
        config = LlamaConfig(vocab_size=4, hidden_size=8, intermediate_size=16,
                             num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2)
        def initialized():
            set_seed(42)
            return get_peft_model(LlamaForCausalLM(config), LoraConfig(
                r=4, lora_alpha=8, target_modules=["q_proj"], task_type="CAUSAL_LM"))
        first, second = initialized(), initialized()
        before = {n: p.detach().clone() for n,p in first.named_parameters() if p.requires_grad}
        self.assertTrue(all(torch.equal(p, before[n]) for n,p in second.named_parameters() if n in before))
        dataset = Dataset.from_list([{"input_ids": [2, 2, 3, 3], "completion_mask": [0, 0, 1, 1]}]*64)
        threads = torch.get_num_threads()
        try:
            torch.set_num_threads(1)
            with tempfile.TemporaryDirectory() as tmp:
                args = replace(training_config(tmp, smoke=True, bf16=False),
                    use_cpu=True, fp16=False, optim="adamw_torch", gradient_checkpointing=False,
                    disable_tqdm=True)
                trainer = SFTTrainer(model=first, args=args, train_dataset=dataset, processing_class=tokenizer)
                result = trainer.train()
                self.assertEqual(trainer.state.global_step, 2)
                self.assertEqual(trainer.state.epoch, 1.0)
                self.assertTrue(torch.isfinite(torch.tensor(result.training_loss)))
                self.assertTrue(any(not torch.equal(p, before[n]) for n,p in first.named_parameters() if n in before))
        finally:
            torch.set_num_threads(threads)

class ApiContractTests(unittest.TestCase):
    def test_readiness_bounds_and_safe_errors(self):
        import asyncio
        import httpx
        from unittest.mock import MagicMock
        import app as api
        # Avoid TestClient's cross-thread portal in the restricted sandbox.
        # ASGI requests/lifespan are real; only sync endpoint dispatch is inline.
        async def inline(function, *args, **kwargs):
            return function(*args, **kwargs)
        async def exercise():
            with patch.dict(os.environ, {"MATH_ADAPTER_PATH": "/ignored/fixture"}), \
                 patch.object(api, "MathSolver", side_effect=RuntimeError("private details")), \
                 patch("fastapi.routing.run_in_threadpool", inline):
                async with api.lifespan(api.app), httpx.AsyncClient(
                        transport=httpx.ASGITransport(app=api.app), base_url="http://test") as client:
                    self.assertEqual((await client.get("/")).status_code, 503)
                    self.assertEqual((await client.post("/solve", json={"question": "Q"})).status_code, 503)
            solver = MagicMock()
            solver.solve.side_effect = lambda **kw: "4" if api.app.state.generation_lock.locked() else self.fail("Generation lock was not held")
            with patch.dict(os.environ, {"MATH_ADAPTER_PATH": "/ignored/fixture"}), \
                 patch.object(api, "MathSolver", return_value=solver), \
                 patch("fastapi.routing.run_in_threadpool", inline):
                async with api.lifespan(api.app), httpx.AsyncClient(
                        transport=httpx.ASGITransport(app=api.app), base_url="http://test") as client:
                    self.assertEqual((await client.get("/")).json(), {"status": "ready"})
                    for invalid in ({"question": " "}, {"question": "Q", "max_new_tokens": 0},
                                    {"question": "Q", "max_new_tokens": 513},
                                    {"question": "Q", "temperature": 0}):
                        self.assertEqual((await client.post("/solve", json=invalid)).status_code, 422)
                    response = await client.post("/solve", json={"question": "2+2?"})
                    self.assertEqual(response.status_code, 200)
                    self.assertEqual(response.json()["answer"], "4")
                    solver.solve.side_effect = RuntimeError("private details")
                    response = await client.post("/solve", json={"question": "Q"})
                    self.assertEqual(response.status_code, 500)
                    self.assertNotIn("private", response.text)
            self.assertIsNone(api.app.state.solver)
        asyncio.run(exercise())
