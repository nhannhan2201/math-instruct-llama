"""Run repository GPU smokes, verify MLflow data, and export evidence.

Usage: python -m src.smoke_review
Metrics displayed and exported come from MlflowClient, not parsed Trainer stdout.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sqlite3
import subprocess
import sys
import zipfile
from datetime import datetime, timezone
from urllib.parse import urlparse, unquote

from mlflow.tracking import MlflowClient


def git(*args):
    return subprocess.check_output(["git", *args], cwd=REPO, text=True).strip()


def run_logged(command, name, env=None):
    log = SESSION / (name + ".log")
    with log.open("w") as stream:
        process = subprocess.Popen(command, cwd=REPO, env=env, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True, bufsize=1)
        for line in process.stdout:
            print(line, end="", flush=True)
            stream.write(line); stream.flush()
        returncode = process.wait()
    if returncode:
        raise RuntimeError(f"{name} failed ({returncode}); see {log}")
    return log.read_text()

RUNS = {}
def smoke(method):
    assert method not in RUNS, "Already run in this notebook session"
    assert git("rev-parse", "HEAD") == HEAD and not git("status", "--porcelain"), "Source changed"
    output = run_logged([sys.executable, "-m", "src.train", "--method", method, "--smoke"], method + "-train", ENV)
    paths = [line.split("Saved adapter/tokenizer/protocol:", 1)[1].strip()
             for line in output.splitlines() if line.startswith("Saved adapter/tokenizer/protocol:")]
    assert len(paths) == 1, "Missing/ambiguous saved artifact path"
    final = (REPO / paths[0]).resolve()
    assert final.is_relative_to(REPO / "models") and final.is_dir()
    metadata = json.loads((final / "effective_data.json").read_text())
    assert metadata['method'] == method and metadata['training_selection']['mode'] == 'smoke'
    RUNS[method] = {"final": final, "metadata": metadata, "run_id": metadata['run_id']}
    run_logged([sys.executable, "-m", "src.inference", "--method", method,
                "--adapter-path", str(final), "--greedy"], method + "-reload", ENV)

def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""): h.update(block)
    return h.hexdigest()
def verify(method, info):
    meta = info['metadata']; run = client.get_run(info['run_id'])
    assert run.info.status == 'FINISHED', run.info.status
    assert run.data.params['method'] == method and run.data.params['mode'] == 'smoke'
    for param, value in {'base_revision': meta['base_revision'], 'dataset_revision': meta['revision'],
                         'precision': meta['precision'], 'train_samples': 64, 'max_steps': 2}.items():
        assert run.data.params[param] == str(value), param
    assert meta['training_selection']['count'] == 64
    assert meta['counts']['validation'] == 100
    for key in ('loss', 'grad_norm', 'adapter_grad_norm'):
        history = client.get_metric_history(info['run_id'], key)
        assert len(history) == 2 and all(math.isfinite(x.value) for x in history), key
        assert {x.step for x in history} == ({0, 1} if key == 'adapter_grad_norm' else {1, 2}), key
    required = ('final_eval_loss', 'adapter_update_norm', 'train_and_eval_seconds',
                'peak_allocated_bytes', 'peak_reserved_bytes', 'trainable_parameters')
    for key in required:
        assert key in run.data.metrics and math.isfinite(run.data.metrics[key]), key
        if key != 'final_eval_loss': assert run.data.metrics[key] > 0, key
    assert run.data.metrics['final_eval_loss'] >= 0
    assert run.data.metrics['peak_reserved_bytes'] >= run.data.metrics['peak_allocated_bytes']
    destination = SESSION / (method + '-download'); destination.mkdir(exist_ok=True)
    downloaded = Path(client.download_artifacts(info['run_id'], 'final', str(destination)))
    top = Path(client.download_artifacts(info['run_id'], 'effective_data.json', str(destination)))
    assert json.loads(top.read_text()) == meta
    assert json.loads((downloaded / 'effective_data.json').read_text()) == meta
    names = {str(p.relative_to(info['final'])) for p in info['final'].rglob('*') if p.is_file()}
    assert 'adapter_config.json' in names and 'adapter_model.safetensors' in names
    assert 'tokenizer_config.json' in names and 'tokenizer.json' in names
    assert names == {str(p.relative_to(downloaded)) for p in downloaded.rglob('*') if p.is_file()}
    hashes = {name: sha(info['final'] / name) for name in sorted(names)}
    assert all(sha(downloaded / name) == value for name, value in hashes.items())
    return {'method': method, 'status': 'PASS', 'run_id': info['run_id'],
            'artifact_uri': run.info.artifact_uri, 'metrics': {k: run.data.metrics[k] for k in required},
            'sha256': hashes}


def main(argv=None):
    global REPO, HEAD, SESSION, ENV, RUNS, DB, client
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=Path('/kaggle/working'))
    args = parser.parse_args(argv)
    REPO = Path(__file__).resolve().parents[1]
    HEAD = git('rev-parse', 'HEAD')
    if git('status', '--porcelain'):
        raise RuntimeError('Working tree must be clean; commit/push source before Kaggle use')
    SESSION = args.output_dir.resolve() / ('smoke-review-' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    SESSION.mkdir(parents=True)
    (SESSION / 'source.json').write_text(json.dumps({'url': git('remote','get-url','origin'), 'head': HEAD}, indent=2))
    ENV = dict(os.environ, CUDA_VISIBLE_DEVICES='0', PYTHONUNBUFFERED='1')
    RUNS = {}
    preflight = """
    import json, torch
    import transformers, trl, peft, accelerate, datasets, bitsandbytes, mlflow, huggingface_hub
    from importlib.metadata import version
    assert torch.cuda.is_available() and torch.cuda.device_count() == 1, 'Need one visible CUDA GPU'
    print(json.dumps({'dependencies': {p: version(p) for p in ('torch','transformers','trl','peft','accelerate','datasets','bitsandbytes','mlflow','huggingface_hub')}, 'gpu': torch.cuda.get_device_name(0), 'cuda': torch.version.cuda, 'native_bf16': torch.cuda.is_bf16_supported(including_emulation=False)}, indent=2))
    """
    run_logged([sys.executable, "-c", preflight], "preflight", ENV)
    for method in ('lora', 'qlora'):
        smoke(method)
    DB = REPO / 'mlruns.db'
    if not DB.is_file() or DB.stat().st_size == 0:
        raise RuntimeError('Missing MLflow database; refusing to create an empty store')
    client = MlflowClient(tracking_uri='sqlite:///' + str(DB))
    report = {'head': HEAD, 'runs': [], 'errors': []}
    print('method | status | run_id | final_eval_loss | seconds | peak_allocated_bytes')
    for method, info in RUNS.items():
        try:
            row = verify(method, info)
            report['runs'].append(row)
            m = row['metrics']
            print(f"{method} | PASS | {row['run_id']} | {m['final_eval_loss']} | {m['train_and_eval_seconds']} | {m['peak_allocated_bytes']}")
            print('MLflow artifact URI:', row['artifact_uri'])
        except Exception as error:
            report['errors'].append({'method': method, 'error': repr(error)})
            print(method, 'FAIL', repr(error))
    for key in ('revision','snapshot_hash','manifest_hash','counts','effective_ids_hash',
                'token_lengths_hash','tokenizer','training_selection'):
        if RUNS['lora']['metadata'][key] != RUNS['qlora']['metadata'][key]:
            report['errors'].append({'parity': key})
    (SESSION / 'verification.json').write_text(json.dumps(report, indent=2))
    # Export evidence even when verification checks fail.
    EXPORT = SESSION / 'export'; EXPORT.mkdir(exist_ok=False)
    with sqlite3.connect('file:' + str(DB) + '?mode=ro', uri=True) as source:
        with sqlite3.connect(str(EXPORT / 'mlruns.db')) as target:
            source.backup(target)
            assert target.execute('PRAGMA integrity_check').fetchone()[0] == 'ok'
    # Copy full local MLflow artifact store (not just adapter).
    assert (REPO / 'mlruns').is_dir(), 'Expected mlruns artifact store missing; inspect artifact_uri'
    shutil.copytree(REPO / 'mlruns', EXPORT / 'mlruns')
    for method, info in RUNS.items():
        uri = urlparse(client.get_run(info['run_id']).info.artifact_uri)
        assert uri.scheme in ('', 'file'), 'Remote artifact store requires separate export'
        artifact_root = Path(unquote(uri.path)).resolve()
        assert artifact_root.is_relative_to((REPO / 'mlruns').resolve()), 'Artifact store outside mlruns; export manually'
        shutil.copytree(info['final'].parent, EXPORT / 'models' / info['final'].parent.name)
    shutil.copy2(REPO / 'data/split_manifest.json', EXPORT / 'split_manifest.json')
    for item in SESSION.iterdir():
        if item.is_file(): shutil.copy2(item, EXPORT / item.name)
    (EXPORT / 'RESTORE.txt').write_text('SQLite retains original Kaggle artifact URIs. Restore at the original path or explicitly relocate artifact URIs before using MLflow on another host. Export contains the artifact files.\n')
    files = {str(p.relative_to(EXPORT)): sha(p) for p in EXPORT.rglob('*') if p.is_file()}
    (EXPORT / 'checksums.json').write_text(json.dumps(files, indent=2))
    ARCHIVE = SESSION.with_suffix('.zip')
    with zipfile.ZipFile(ARCHIVE, 'w', zipfile.ZIP_DEFLATED) as z:
        for p in EXPORT.rglob('*'):
            if p.is_file(): z.write(p, str(p.relative_to(EXPORT)))
    with zipfile.ZipFile(ARCHIVE) as z:
        assert z.testzip() is None
        assert set(z.namelist()) == set(files) | {'checksums.json'}
        for name, expected in files.items():
            h = hashlib.sha256()
            with z.open(name) as f:
                for block in iter(lambda: f.read(1024 * 1024), b''): h.update(block)
            assert h.hexdigest() == expected, name
    print('ZIP verified:', ARCHIVE, 'SHA256:', sha(ARCHIVE))
    print('Download this ZIP from Kaggle Output before reset.')
    if report['errors']:
        raise RuntimeError('MLflow verification FAIL; exported evidence for debugging')
    print('MLflow verification PASS; greedy reload is not accuracy or numerical tolerance proof.')


if __name__ == '__main__':
    main()
