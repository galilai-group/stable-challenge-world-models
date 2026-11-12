"""CPU/GPU evaluation of an ONNX action-conditioned world model, with linear probes and latent planning.

    python evaluate.py model.onnx path/to/eval_data [--out results.json]

The script is self-contained so it can be uploaded as a challenge evaluator. ``eval_data`` is the unzipped
evaluation data (a folder with ``tasks.json``); pass it as an absolute path.

Problems with the submitted model are reported as {"error": ...} on stdout, which the leaderboard shows.
Prints {"score": <float>, "score_<task>": <float>, ...} as JSON on stdout; the detailed report goes to stderr.

The model contract (see CHALLENGE.md):
    inputs   frames   float32 [64, HIST, 3, 64, 64]   the last HIST observations, SKIP env steps apart, in [0, 1]
             actions  float32 [64, HORIZON, 2*SKIP]   the next HORIZON*SKIP env actions, grouped SKIP per step
    output   z        float32 [64, HORIZON + 1, D]    z[:, 0] embeds the current observation (it must not depend
                                                      on `actions`); z[:, k] is the predicted embedding k steps
                                                      (k*SKIP env steps) later, using only actions[:, :k]

Each environment in the data contributes some of these tasks (listed in tasks.json):
    position  fit a ridge probe from z[:, 0] of real observations to the agent's (x, y) position, then apply it
              to the *predicted* embeddings z[:, 1..HORIZON] of held-out rollouts; metric = mean over horizons of
              R^2 clipped at 0
    velocity  the same for the agent's velocity (proprioception; environments that record it)
    inverse   inverse dynamics: a ridge probe from the real embeddings of two windows SKIP steps apart to the mean
              action taken in between; metric = held-out R^2 clipped at 0
    plan      given a start and a goal window, rank CANDIDATES action sequences (the true one; the others mostly
              the true one with 1-2 steps swapped, some from other episodes) by the distance between the predicted
              final embedding and the goal's embedding; metric = top-1 accuracy (ties split evenly), chance = 1/8
    task score = max((metric - floor) / (reference - floor), 0): 0 at an untrained encoder with no dynamics, 1 at
    the organizers' baseline. Score = weighted mean of the task scores (weights and anchors come with the data).

Evaluation data layout, per environment folder ``<env>/``:
    fit_frames   uint8 [N, HIST, 64, 64, 3]     fit_pos   float32 [N, 2]   fit_vel float32 [N, 2]
    roll_frames  uint8 [M, HIST, 64, 64, 3]     roll_actions float32 [M, HORIZON, 2*SKIP]
                 roll_pos float32 [M, HORIZON, 2]   roll_vel float32 [M, HORIZON, 2]
    inv_fit_frames, inv_test_frames   uint8 [I, HIST + 1, 64, 64, 3]   (the windows at t and t + SKIP)
    inv_fit_actions, inv_test_actions float32 [I, 2]                   (the mean of a[t], ..., a[t+SKIP-1])
    fit_vel, roll_vel only for environments with a velocity task
    plan_frames  uint8 [P, HIST, 64, 64, 3]     plan_goal uint8 [P, HIST, 64, 64, 3]   (the HIST observations up to the goal)
    plan_actions float32 [P, CANDIDATES, HORIZON, 2*SKIP]    plan_answer int64 [P]
"""

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import onnxruntime as ort
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score

BATCH, HIST, SKIP, HORIZON, SIZE, ACTION_DIM = 64, 3, 5, 5, 64, 2
MAX_DIM = 1024
ALPHAS = np.logspace(-2, 4, 13)  # ridge penalty, chosen by leave-one-out on the fitted items
FRAMES_SHAPE = (BATCH, HIST, 3, SIZE, SIZE)
ACTIONS_SHAPE = (BATCH, HORIZON, ACTION_DIM * SKIP)
CUDA, CPU = "CUDAExecutionProvider", "CPUExecutionProvider"


class SubmissionError(Exception):
    """A problem with the submitted model, shown to the participant."""


def load_model(path):
    if hasattr(ort, "preload_dlls"):
        try:
            ort.preload_dlls()
        except Exception:
            pass
    opts = ort.SessionOptions()
    opts.log_severity_level = 3
    providers = [CUDA, CPU] if CUDA in ort.get_available_providers() else [CPU]
    try:
        sess = ort.InferenceSession(path, opts, providers=providers)
    except Exception as e:
        raise SubmissionError(f"Could not load the ONNX model: {e}") from None
    if sess.get_providers()[0] != CUDA:
        # one single-threaded batch per core in parallel (see run_batches)
        opts.intra_op_num_threads = 1
        sess = ort.InferenceSession(path, opts, providers=[CPU])
    ins, outs = sess.get_inputs(), sess.get_outputs()
    if len(ins) != 2 or len(outs) != 1:
        raise SubmissionError("ONNX model must have exactly two inputs (frames, actions) and one output (z)")
    names = sorted(i.name for i in ins)
    if names != ["actions", "frames"]:
        raise SubmissionError(f"ONNX inputs must be named 'frames' and 'actions'; got {names}")
    for i in ins:
        want = FRAMES_SHAPE if i.name == "frames" else ACTIONS_SHAPE
        if i.type != "tensor(float)":
            raise SubmissionError(f"Input '{i.name}' must be float32, not {i.type}")
        if len(i.shape) != len(want) or any(isinstance(g, int) and g != w for g, w in zip(i.shape, want)):
            raise SubmissionError(f"Input '{i.name}' must accept shape {want}; it declares {i.shape}")
    return sess


def to_input(frames):
    """uint8 [n, HIST, 64, 64, 3] -> float32 [n, HIST, 3, 64, 64] in [0, 1]."""
    return np.ascontiguousarray(np.asarray(frames, np.float32).transpose(0, 1, 4, 2, 3) / 255.0)


def run_batches(sess, frames, actions):
    """frames uint8 [n, HIST, 64, 64, 3], actions float32 [n, HORIZON, 2*SKIP] -> z float32 [n, HORIZON+1, D].

    Always runs full batches of BATCH; the last one is zero-padded and trimmed. On CPU, one batch per core.
    """
    n = len(frames)

    def run(i):
        f = np.zeros(FRAMES_SHAPE, np.float32)
        a = np.zeros(ACTIONS_SHAPE, np.float32)
        m = min(BATCH, n - i)
        f[:m] = to_input(frames[i : i + m])
        a[:m] = actions[i : i + m]
        try:
            z = sess.run(None, {"frames": f, "actions": a})[0]
        except Exception as e:
            raise SubmissionError(f"Model failed on frames {FRAMES_SHAPE} and actions {ACTIONS_SHAPE}: {e}") from None
        if not isinstance(z, np.ndarray) or z.ndim != 3 or z.shape[:2] != (BATCH, HORIZON + 1):
            raise SubmissionError(f"Output shape {getattr(z, 'shape', None)}, expected ({BATCH}, {HORIZON + 1}, D)")
        return z[:m]

    starts = range(0, n, BATCH)
    if sess.get_providers()[0] == CPU:
        cores = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count()
        with ThreadPoolExecutor(cores) as pool:
            out = list(pool.map(run, starts))
    else:
        out = [run(i) for i in starts]
    dims = {z.shape[2] for z in out}
    if len(dims) != 1:
        raise SubmissionError("Model returned inconsistent embedding widths")
    dim = dims.pop()
    if not 1 <= dim <= MAX_DIM:
        raise SubmissionError(f"Embedding width {dim}, expected between 1 and {MAX_DIM}")
    z = np.concatenate(out).astype(np.float32)
    if not np.isfinite(z).all():
        raise SubmissionError("Embeddings contain NaN or infinite values")
    return z


def encode(sess, frames):
    """Embeddings z[:, 0] of real observations (actions are zero and must not matter)."""
    return run_batches(sess, frames, np.zeros((len(frames),) + ACTIONS_SHAPE[1:], np.float32))[:, 0]


def cached(d, key, fn):
    """Embeddings shared by several tasks of one environment are computed once."""
    if key not in d:
        d[key] = fn()
    return d[key]


def fit_embeddings(sess, d):
    """z[:, 0] of the probe-fitting observations, with one global center and scale (not per dimension, which would
    blow up near-constant dimensions). Returns (x_fit, normalize)."""
    def make():
        x = encode(sess, d["fit_frames"]).astype(np.float64)
        mu, sd = x.mean(0), x.std() + 1e-12
        return (x - mu) / sd, lambda z: (np.asarray(z, np.float64) - mu) / sd
    return cached(d, "_fit", make)


def rollout_r2(sess, d, target):
    """Probe fit on real embeddings -> `target`, applied to predicted ones: mean R^2 over the horizons."""
    x_fit, norm = fit_embeddings(sess, d)
    probe = RidgeCV(alphas=ALPHAS).fit(x_fit, d[f"fit_{target}"])
    z = cached(d, "_roll", lambda: run_batches(sess, d["roll_frames"], d["roll_actions"]))
    r2 = []
    for k in range(1, HORIZON + 1):
        pred = probe.predict(norm(z[:, k]))
        r2.append(max(r2_score(d[f"roll_{target}"][:, k - 1], pred), 0.0))  # worse than the mean counts as 0
    return float(np.mean(r2)), {"r2_per_horizon": [round(v, 4) for v in r2]}


def position_r2(sess, d):
    """Predicted embeddings -> the agent's (x, y) position."""
    return rollout_r2(sess, d, "pos")


def velocity_r2(sess, d):
    """Predicted embeddings -> the agent's velocity (proprioception)."""
    return rollout_r2(sess, d, "vel")


def inverse_r2(sess, d):
    """Inverse dynamics: a ridge probe from the real embeddings of two windows SKIP steps apart to the mean action
    taken in between; R^2 on held-out episodes, clipped at 0."""
    def features(frames):  # frames [n, HIST + 1, ...]: the windows at t and t + SKIP
        a, b = encode(sess, frames[:, :HIST]), encode(sess, frames[:, 1:])
        return np.concatenate([a, b - a], 1).astype(np.float64)
    x_fit, x_test = features(d["inv_fit_frames"]), features(d["inv_test_frames"])
    mu, sd = x_fit.mean(0), x_fit.std() + 1e-12
    probe = RidgeCV(alphas=ALPHAS).fit((x_fit - mu) / sd, d["inv_fit_actions"])
    r2 = r2_score(d["inv_test_actions"], probe.predict((x_test - mu) / sd))
    return max(float(r2), 0.0), {}


def plan_accuracy(sess, d):
    """Rank candidate action sequences by ||z_pred[HORIZON] - z_goal||^2; top-1 accuracy, ties split evenly."""
    frames, acts, answer = d["plan_frames"], d["plan_actions"], d["plan_answer"]
    p, c = acts.shape[:2]
    goal = encode(sess, d["plan_goal"]).astype(np.float64)
    z = run_batches(sess, np.repeat(frames, c, axis=0), acts.reshape(p * c, *acts.shape[2:]))
    final = z[:, HORIZON].astype(np.float64).reshape(p, c, -1)
    dist = ((final - goal[:, None]) ** 2).sum(-1)
    best = dist <= dist.min(1, keepdims=True) * (1 + 1e-6) + 1e-12
    hit = best[np.arange(p), answer] / best.sum(1)
    return float(hit.mean()), {"chance": 1.0 / c}


TASKS = {"position": position_r2, "velocity": velocity_r2, "inverse": inverse_r2, "plan": plan_accuracy}


def load_env(folder):
    return {f[:-4]: np.load(os.path.join(folder, f)) for f in os.listdir(folder) if f.endswith(".npy")}


def evaluate(model_path, data):
    t0 = time.time()
    sess = load_model(model_path)
    print(f"running on {sess.get_providers()[0]}", file=sys.stderr, flush=True)
    tasks = json.load(open(os.path.join(data, "tasks.json")))["tasks"]
    envs, res = {}, {}
    for task in tasks:
        env = task["env"]
        if env not in envs:
            envs[env] = load_env(os.path.join(data, env))
        metric, info = TASKS[task["type"]](sess, envs[env])
        # below the floor (no better than no dynamics) counts as 0, so one failed task can't sink the others
        score = max((metric - task["floor"]) / max(task["reference"] - task["floor"], 1e-6), 0.0)
        name = f"{env}_{task['type']}"
        res[name] = {"metric": metric, "score": score, "weight": task["weight"], **info}
        print(f"{name:<20s} metric {metric:.4f}  score {score:7.4f}  [{time.time() - t0:.0f}s]",
              file=sys.stderr, flush=True)
    weights = np.array([r["weight"] for r in res.values()])
    score = float(np.dot(weights, [r["score"] for r in res.values()]) / weights.sum())
    return {"score": score, "tasks": res, "seconds": round(time.time() - t0, 1)}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("model", help="path to the ONNX model")
    ap.add_argument("data", help="directory of the unzipped evaluation data")
    ap.add_argument("--out", default=None, help="also write the full results JSON here")
    a = ap.parse_args()
    if not os.path.isfile(os.path.join(a.data, "tasks.json")):
        sys.exit(f"Evaluation data not found at {a.data!r}")
    try:
        r = evaluate(a.model, a.data)
    except SubmissionError as e:
        print(json.dumps({"error": str(e)}))
        sys.exit(1)
    print(f"score {r['score']:.4f}  time {r['seconds']}s", file=sys.stderr)
    if a.out:
        with open(a.out, "w") as f:
            json.dump(r, f, indent=2)
    print(json.dumps({"score": round(r["score"], 6),
                      **{f"score_{k}": round(v["score"], 6) for k, v in r["tasks"].items()}}))
