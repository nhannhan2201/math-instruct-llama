"""CLI demo regression, no GPU or pretrained model downloads."""
import contextlib
import io
import unittest
from unittest.mock import patch
from src import inference


class InferenceDemoTests(unittest.TestCase):
    def test_suite_loads_once_and_uses_same_generation_settings(self):
        with patch.object(inference, 'MathSolver') as cls, contextlib.redirect_stdout(io.StringIO()) as output:
            cls.return_value.solve.return_value = 'demo answer'
            inference.main(['--method','lora','--adapter-path','final','--greedy',
                            '--demo-suite','--max-new-tokens','512'])
            cls.assert_called_once_with(method='lora', adapter_path='final')
            self.assertEqual(cls.return_value.solve.call_count, 5)
            for call in cls.return_value.solve.call_args_list:
                self.assertEqual(call.kwargs, {'max_new_tokens':512, 'do_sample':False})
            self.assertIn('10/21', output.getvalue())
            self.assertIn('no accuracy score', output.getvalue())

    def test_original_single_question_default_preserved(self):
        with patch.object(inference, 'MathSolver') as cls, contextlib.redirect_stdout(io.StringIO()):
            inference.main(['--adapter-path','final'])
            cls.return_value.solve.assert_called_once_with(inference.DEMO_QUESTIONS[0][1], max_new_tokens=150, do_sample=True)

    def test_invalid_token_limit_fails_before_load(self):
        with patch.object(inference, 'MathSolver') as cls, contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                inference.main(['--adapter-path','final','--max-new-tokens','0'])
            cls.assert_not_called()
