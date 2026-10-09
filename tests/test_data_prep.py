import copy
import json
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

from datasets import Dataset
from src.data_prep import (allocate_quota, build_manifest, canonical_json,
                           format_batch, load_or_create_manifest, prepared_from_manifest, prepare_data)


class DataPreparationTests(unittest.TestCase):
    def setUp(self):
        self.rows = [{"instruction": f"Question {i} + 1?", "output": str(i + 1),
                      "source": f"data/{'CoT' if i % 2 else 'PoT'}/sample.json"} for i in range(40)]
        self.rows += [dict(self.rows[0]), {**self.rows[0], "instruction": " Question  0 + 1? ",
                                         "output": " 1 "},
                      {**self.rows[0], "output": "An alternative solution: 1"}]
        self.raw = Dataset.from_list(self.rows)
        self.settings = dict(validation_size=3, test_size=5, train_fraction=.3)

    def build(self, **kwargs):
        return build_manifest(self.raw, "test-revision", **{**self.settings, **kwargs})

    def test_dedup_distinct_outputs_and_symbols(self):
        m = self.build()
        self.assertEqual(len(m['pairs']), 41)
        self.assertEqual(len(m['groups']), 40)
        pair = next(p for p in m['pairs'].values() if 0 in p['raw_indices'])
        self.assertEqual(pair['raw_indices'], [0, 40, 41])
        group = m['groups'][pair['group_id']]
        self.assertEqual(len(group['pair_ids']), 2)
        changed = Dataset.from_list(self.rows + [{**self.rows[0], 'instruction': 'Question 0 - 1?'}])
        self.assertEqual(len(build_manifest(changed, 'test', **self.settings)['groups']), 41)

    def test_quotas(self):
        self.assertEqual(allocate_quota({'b': 1, 'a': 1}, 1), {'b': 0, 'a': 1})
        self.assertEqual(allocate_quota({'b': 1, 'a': 1}, 1), allocate_quota({'a': 1, 'b': 1}, 1))
        for n in range(11):
            quota = allocate_quota({'a': 2, 'b': 8}, n)
            self.assertEqual(sum(quota.values()), n)
            self.assertLessEqual(quota['a'], 2)
        with self.assertRaises(ValueError):
            allocate_quota({'a': 2}, 3)

    def test_deterministic_and_zero_overlap(self):
        a, b = self.build(), self.build()
        self.assertEqual(canonical_json(a), canonical_json(b))
        splits = {name: {gid for gid, g in a['groups'].items() if g['split'] == name}
                  for name in ('train', 'validation', 'test')}
        self.assertEqual(len(splits['validation']), 3)
        self.assertEqual(len(splits['test']), 5)
        self.assertFalse(splits['train'] & splits['validation'])
        self.assertFalse(splits['train'] & splits['test'])
        self.assertFalse(splits['validation'] & splits['test'])
        self.assertEqual(set.union(*splits.values()), set(a['groups']))
        pool = [pid for pid, p in a['pairs'].items() if p['group_id'] in splits['train']]
        self.assertEqual(len(a['training_subset']), int(.3 * len(pool)))
        self.assertTrue(set(a['training_subset']) <= set(pool))
        self.assertNotEqual(a['training_subset'], self.build(seed=43)['training_subset'])

    def test_references(self):
        m = self.build()
        for split, references in m['references'].items():
            for ref in references:
                g = m['groups'][ref['group_id']]
                expected = sorted(m['pairs'][pid]['raw_indices'][0] for pid in g['pair_ids'])
                self.assertEqual(ref['all_references'], expected)
                self.assertEqual(ref['primary_reference'], min(expected))
                self.assertEqual(g['split'], split)

    def test_manifest_integrity_snapshot_and_settings(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'manifest.json'
            m = load_or_create_manifest(self.raw, 'test-revision', path, **self.settings)
            original = path.read_bytes()
            self.assertEqual(m, load_or_create_manifest(self.raw, 'test-revision', path, **self.settings))
            self.assertEqual(original, path.read_bytes())
            changed = copy.deepcopy(self.rows)
            changed[0]['output'] = 'corrupted'
            with self.assertRaisesRegex(ValueError, 'snapshot'):
                load_or_create_manifest(Dataset.from_list(list(reversed(self.rows))), 'test-revision', path, **self.settings)
            with self.assertRaisesRegex(ValueError, 'snapshot'):
                load_or_create_manifest(Dataset.from_list(changed), 'test-revision', path, **self.settings)
            with self.assertRaisesRegex(ValueError, 'snapshot'):
                load_or_create_manifest(self.raw, 'other-revision', path, **self.settings)
            with self.assertRaisesRegex(ValueError, 'integrity/settings'):
                load_or_create_manifest(self.raw, 'test-revision', path, **{**self.settings, 'seed': 43})
            m['references']['test'][0]['primary_reference'] = 99999
            path.write_text(json.dumps(m))
            with self.assertRaisesRegex(ValueError, 'integrity/settings'):
                load_or_create_manifest(self.raw, 'test-revision', path, **self.settings)

    def test_missing_source_metadata_requires_pinned_content(self):
        from src.config import DATASET_REVISION
        from src.data_prep import cached_revision
        self.raw.info.download_checksums = None
        revision = cached_revision(self.raw)
        self.assertEqual(revision, DATASET_REVISION)
        expected = build_manifest(self.raw, revision, **self.settings)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp, "manifest.json")
            # Fixture content identity stands in for the verified real snapshot.
            with patch("src.data_prep.DATASET_SNAPSHOT_HASH", expected["snapshot_hash"]):
                actual = load_or_create_manifest(self.raw, revision, path, **self.settings)
                self.assertEqual(actual, expected)
                original = path.read_bytes()
                changed = copy.deepcopy(self.rows)
                changed[0]["output"] = "tampered"
                with self.assertRaisesRegex(ValueError, "pinned snapshot"):
                    load_or_create_manifest(Dataset.from_list(changed), revision, path, **self.settings)
                self.assertEqual(path.read_bytes(), original)
                absent = Path(tmp, "wrong.json")
                with self.assertRaisesRegex(ValueError, "pinned snapshot"):
                    load_or_create_manifest(Dataset.from_list(changed), revision, absent, **self.settings)
                self.assertFalse(absent.exists())

    def test_training_entrypoint_compatible(self):
        from transformers import AutoTokenizer
        from src.config import BASE_MODEL, TOKENIZER_REVISION
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, revision=TOKENIZER_REVISION, local_files_only=True)
        m = self.build()
        with patch('src.data_prep.load_dataset', return_value={'train': self.raw}), \
             patch('src.data_prep.cached_revision', return_value='test-revision'), \
             patch('src.data_prep.load_or_create_manifest', return_value=m):
            train, evaluation = prepare_data(tokenizer)
        self.assertEqual(train.column_names, ['prompt', 'completion', 'chat_template_kwargs'])
        self.assertEqual(evaluation.column_names, ['prompt', 'completion', 'chat_template_kwargs'])
        self.assertEqual(len(evaluation), 3)

    def test_schema_and_original_formatter(self):
        from transformers import AutoTokenizer
        from src.config import BASE_MODEL, TOKENIZER_REVISION
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, revision=TOKENIZER_REVISION, local_files_only=True)
        m = self.build()
        train, validation = prepared_from_manifest(self.raw, tokenizer, m)
        self.assertEqual(train.column_names, ['prompt', 'completion', 'chat_template_kwargs'])
        self.assertEqual(validation.column_names, ['prompt', 'completion', 'chat_template_kwargs'])
        i = m['pairs'][m['training_subset'][0]]['raw_indices'][0]
        r = self.raw[i]
        expected = format_batch({'instruction': [r['instruction']], 'output': [r['output']]}, tokenizer)
        self.assertEqual(train[0], {key: values[0] for key, values in expected.items()})


if __name__ == '__main__':
    unittest.main()
