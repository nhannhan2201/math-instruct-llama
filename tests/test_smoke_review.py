"""Actual offline SQLite MLflow verification, without models or GPU."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from mlflow.tracking import MlflowClient
from src import smoke_review as review


class SmokeReviewTests(unittest.TestCase):
    def test_preflight_python_c_source_compiles(self):
        # Exercise main's actual command construction, not a duplicate snippet.
        class StopAfterPreflight(Exception):
            pass

        def capture(command, name, env):
            self.assertEqual(name, 'preflight')
            self.assertEqual(command[1], '-c')
            compile(command[2], '<preflight>', 'exec')
            self.assertIn('including_emulation=False', command[2])
            raise StopAfterPreflight

        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(review, 'git', side_effect=lambda *args: '' if args[0] == 'status' else 'test-head'), patch.object(review, 'run_logged', side_effect=capture):
                with self.assertRaises(StopAfterPreflight):
                    review.main(['--output-dir', tmp])

    def test_metrics_and_artifacts_from_real_mlflow(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            client = MlflowClient(tracking_uri=f'sqlite:///{root}/tracking.db')
            experiment = client.create_experiment('smoke-test', artifact_location=(root / 'mlruns').as_uri())
            run = client.create_run(experiment)
            rid = run.info.run_id
            final = root / 'final'
            final.mkdir()
            meta = {'method': 'lora', 'run_id': rid, 'base_revision': 'base',
                    'revision': 'data', 'precision': 'torch.float16',
                    'training_selection': {'count': 64}, 'counts': {'validation': 100}}
            for name in ('adapter_config.json', 'adapter_model.safetensors',
                         'tokenizer_config.json', 'tokenizer.json'):
                (final / name).write_text('{}')
            (final / 'effective_data.json').write_text(json.dumps(meta))
            for key, value in {'method': 'lora', 'mode': 'smoke', 'base_revision': 'base',
                               'dataset_revision': 'data', 'precision': 'torch.float16',
                               'train_samples': '64', 'max_steps': '2'}.items():
                client.log_param(rid, key, value)
            for key in ('loss', 'grad_norm', 'adapter_grad_norm'):
                for step in ([0, 1] if key == 'adapter_grad_norm' else [1, 2]):
                    client.log_metric(rid, key, 0.5, step=step)
            for key in ('adapter_update_norm', 'train_and_eval_seconds', 'peak_allocated_bytes',
                        'peak_reserved_bytes', 'trainable_parameters'):
                client.log_metric(rid, key, 1.0)
            client.log_metric(rid, 'final_eval_loss', 1.234)
            client.log_artifacts(rid, str(final), 'final')
            client.log_artifact(rid, str(final / 'effective_data.json'))
            client.set_terminated(rid)
            info = {'metadata': meta, 'run_id': rid, 'final': final}
            with patch.object(review, 'client', client, create=True), patch.object(review, 'SESSION', root, create=True):
                row = review.verify('lora', info)
                self.assertEqual(row['metrics']['final_eval_loss'], 1.234)
                self.assertEqual(row['run_id'], rid)
                client.log_metric(rid, 'final_eval_loss', 2.345, step=3)
                self.assertEqual(review.verify('lora', info)['metrics']['final_eval_loss'], 2.345)
                client.log_metric(rid, 'adapter_update_norm', 0, step=3)
                with self.assertRaises(AssertionError):
                    review.verify('lora', info)
                client.log_metric(rid, 'adapter_update_norm', 1, step=4)
                client.set_terminated(rid, status='FAILED')
                with self.assertRaises(AssertionError):
                    review.verify('lora', info)
                client.set_terminated(rid)
                (final / 'tokenizer.json').write_text('changed')
                with self.assertRaises(AssertionError):
                    review.verify('lora', info)


if __name__ == '__main__':
    unittest.main()
