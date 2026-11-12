"""Check a submission before uploading it.

    python test_submission.py model.onnx

Runs the evaluator's own loading and inference code (evaluate.py, next to this file) on random inputs, so a model
that passes here loads and runs the same way in the official evaluation. Checks:

    1. loads in onnxruntime: inputs 'frames' (64, 3, 3, 64, 64) and 'actions' (64, 5, 10), one output   evaluator
    2. returns finite (64, 6, D) embeddings, 1 <= D <= 1024                                            evaluator
    3. file size <= 20 MiB, ONNX opset >= 17
    4. deterministic: the same input gives the same output
    5. each sample is processed independently of the others in its batch
    6. z[:, 0] (the embedding of the observation) does not depend on the actions
    7. causal: z[:, k] does not depend on actions[:, k:] (the actions after step k)
    8. (warning only) the predictions z[:, 1:] do depend on the actions
    9. throughput on this machine, and the full evaluation's time at that rate

Exits 0 if every check passes, 1 otherwise.
"""

import argparse
import os
import sys
import time

import numpy as np

from evaluate import ACTIONS_SHAPE, BATCH, HIST, HORIZON, SIZE, SubmissionError, load_model, run_batches

OPSET_MIN, MAX_MIB = 17, 20
SUITE_SAMPLES = 33_792  # model inputs processed by the full evaluation (both environments)
rng = np.random.default_rng(0)


def report(ok, message):
    print(f"[{' ok ' if ok else 'FAIL'}] {message}")
    return ok


def opset(path):
    try:
        import onnx
    except ImportError:
        return None
    model = onnx.load(path, load_external_data=False)
    return max((i.version for i in model.opset_import if i.domain in ("", "ai.onnx")), default=None)


def random_inputs(n=BATCH):
    frames = rng.integers(0, 256, (n, HIST, SIZE, SIZE, 3), dtype=np.uint8)
    actions = rng.uniform(-1, 1, (n,) + ACTIONS_SHAPE[1:]).astype(np.float32)
    return frames, actions


def close(a, b):
    return np.allclose(a, b, atol=1e-4, rtol=1e-3)


def check(path):
    try:
        sess = load_model(path)
    except SubmissionError as e:
        return report(False, str(e))
    provider = sess.get_providers()[0]
    report(True, f"loads with inputs 'frames' and 'actions' of the contract's shapes; running on {provider}")

    f, a = random_inputs()
    try:
        z = run_batches(sess, f, a)
    except SubmissionError as e:
        return report(False, str(e))
    ok = report(True, f"output {z.shape}, finite")

    mib = os.path.getsize(path) / 2**20
    ok &= report(mib <= MAX_MIB, f"file size {mib:.1f} MiB (maximum {MAX_MIB})")
    version = opset(path)
    if version is None:
        print("[warn] could not read the opset (pip install onnx to check it)")
    else:
        ok &= report(version >= OPSET_MIN, f"opset {version} (minimum {OPSET_MIN})")

    same = np.array_equal(z, run_batches(sess, f, a))
    ok &= report(same, "deterministic" if same else
                 "not deterministic: the same input gave different outputs (dropout left on at export?)")

    perm = rng.permutation(BATCH)
    indep = close(run_batches(sess, f[perm], a[perm]), z[perm])
    ok &= report(indep, "each sample processed independently of its batch" if indep else
                 "outputs depend on the other samples in the batch (batch norm exported in training mode?)")

    _, a2 = random_inputs()
    z2 = run_batches(sess, f, a2)
    enc = close(z2[:, 0], z[:, 0])
    ok &= report(enc, "z[:, 0] does not depend on the actions" if enc else
                 "z[:, 0] changes with the actions; it must embed the observation alone")

    causal = True
    for k in range(1, HORIZON):
        a3 = a.copy()
        a3[:, k:] = a2[:, k:]
        if not close(run_batches(sess, f, a3)[:, : k + 1], z[:, : k + 1]):
            causal = False
            break
    ok &= report(causal, "causal: z[:, k] uses only actions[:, :k]" if causal else
                 f"not causal: z[:, {k}] changes when actions[:, {k}:] change")

    if close(z2[:, 1:], z[:, 1:]):
        print("[warn] the predictions z[:, 1:] ignore the actions: the model has no action-conditioned dynamics")

    f, a = random_inputs(BATCH * 16)
    run_batches(sess, f[:BATCH], a[:BATCH])
    start = time.perf_counter()
    run_batches(sess, f, a)
    rate = len(f) / (time.perf_counter() - start)
    print(f"[info] {rate:,.0f} samples/s on {provider} ({os.cpu_count()} cores): the full evaluation "
          f"({SUITE_SAMPLES:,} samples) would take ~{SUITE_SAMPLES / rate / 60:.1f} min on this machine")
    return ok


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("model", help="path to the ONNX model")
    ok = check(ap.parse_args().model)
    print("\nAll checks passed." if ok else "\nSome checks failed; fix them before submitting.")
    sys.exit(0 if ok else 1)
