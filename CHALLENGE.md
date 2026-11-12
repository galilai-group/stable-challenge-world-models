Pretrain an action-conditioned world model, self-supervised, on recorded episodes of two simple control
environments. It is scored on private held-out episodes by a hidden set of tasks: reading the agent's state out of
its *predicted* future embeddings, capturing the effect of actions, and telling which actions reach a goal.

> **Submission at a glance**
>
> - **Format:** a single `.onnx` file with two inputs (`frames`, `actions`) and one output. Not a PyTorch
>   checkpoint, zip, or weights without the architecture.
> - **Size cap: 20 MiB per file.** Larger uploads are rejected. Check with `ls -lh model.onnx`.
> - **Limit:** up to 5 submissions per day.
> - **Only your latest submission counts:** the leaderboard uses your most recent successfully evaluated model,
>   not your best. A worse resubmission replaces a better one.
> - **Leaderboard updates:** the leaderboard is refreshed periodically (likely every few days), not immediately
>   after each submission.
> - **Before uploading:** `uv run test_submission.py model.onnx` must pass.

## Quick start

The environment is managed with [uv](https://docs.astral.sh/uv/getting-started/installation/).

```bash
git clone https://github.com/galilai-group/stable-challenge-world-models.git
cd stable-challenge-world-models
uv sync                                                    # create .venv with all dependencies
unzip train.zip -d corpus                                  # the training data (see below)
uv run example_train.py --corpus corpus --out model.onnx   # train and export the reference world model
uv run test_submission.py model.onnx                       # check a model before submitting it
```

## Training data

**Download: `train.zip` (80 MB; 3.1 GB unzipped) from [Google Drive](TODO-LINK).** Unzip it to get one file, `train.npz`, with three
arrays:

| Array | dtype, shape | Content |
| --- | --- | --- |
| `frames` | uint8, [251,600, 64, 64, 3] | RGB observations of every episode, concatenated |
| `actions` | float32, [251,600, 2] | `actions[i]` is the action taken right after observing `frames[i]` |
| `offsets` | int64, [3,333] | episode `e` is rows `offsets[e]:offsets[e+1]` |

That is all there is: 3,332 episodes in random order, of a ball or a dot pushed around by random 2-D actions
(episodes are 50 or 100 steps long), with no states, rewards, labels or environment ids. The data comes from
the PointMaze and Wall environments of [DINO-WM](https://dino-wm.github.io/), resized to 64 x 64.
[corpus_loader.py](corpus_loader.py) memory-maps the file:

```python
import corpus_loader
corpus = corpus_loader.load("corpus")
frames, actions = corpus.episode(0)        # uint8 [T, 64, 64, 3], float32 [T, 2]
```

**Rules.** The corpus is the only training data allowed: no other datasets (including the original DINO-WM
data and simulators), no ground-truth states and no pretrained weights. Top submissions must share their training
code, and we retrain them to confirm the final ranking.

## Model format

An ONNX file (opset 17 or newer). The model is called with batches of 64 and *time is measured in steps of
`SKIP = 5` environment steps*:

| | name | dtype, shape | content |
| --- | --- | --- | --- |
| input | `frames` | float32 `[64, 3, 3, 64, 64]` | the last 3 observations `o[t-10], o[t-5], o[t]`, RGB in [0, 1] |
| input | `actions` | float32 `[64, 5, 10]` | the next 25 environment actions `a[t], ..., a[t+24]`; row `k` holds `a[t+5k], ..., a[t+5k+4]` flattened (x, y, x, y, ...) |
| output | any name | float32 `[64, 6, D]`, `D <= 1024` | `z[:, 0]` embeds the current observation; `z[:, k]` is the predicted embedding of the observation at `t + 5k` |

Requirements, all checked by `test_submission.py`:

- `z[:, 0]` must **not depend on `actions`**: it is how the evaluator embeds real observations.
- The prediction must be **causal**: `z[:, k]` may use only `actions[:, :k]`.
- Every sample is processed independently of the rest of the batch, deterministically (export in eval mode).
- Predicted embeddings `z[:, k]` should live in the same space as `z[:, 0]` of the real future observation:
  the evaluation compares them.

Actions are raw (not normalized); normalize inside your model if you want to.

## Evaluation

Submissions are scored on held-out episodes of both environments by a set of tasks that is **not disclosed**, so
that nobody tunes a model to the evaluation instead of making it a good world model. The tasks test what a world
model is for:

- whether the agent's state can be read out of the **predicted** embeddings `z[:, 1..5]`, several steps ahead;
- whether the embeddings capture the **effect of actions**;
- whether the model can tell which actions lead from a start to a **goal** observation.

Some tasks fit simple linear probes on your embeddings, and some compare embeddings by squared Euclidean distance.
So what matters should be linearly readable from `z`, and a prediction should land close to the embedding of the
observation it predicts.

Each task's score is mapped to `max((metric - floor) / (reference - floor), 0)`. **0** is an untrained encoder with no
dynamics (`example_train.py --floor`), and **1** is the reference model trained by `example_train.py`; retraining it
with another seed scores ~0.96. Each environment counts for half of the total, and good submissions should exceed 1.
The evaluation takes well under a minute on 2 CPU cores for the reference model; `test_submission.py` estimates the
time for yours.

## Example

[example_train.py](example_train.py) trains a small JEPA-style world model with
[stable-pretraining](https://github.com/galilai-group/stable-pretraining). It uses a CNN encoder over the 3 stacked
frames and a residual MLP predictor rolled out in latent space. The loss is multi-step latent prediction (MSE),
plus an InfoNCE term (each prediction must identify its own target in the batch) and VCReg against collapse. There
is no pixel reconstruction. It trains in ~7 minutes on a laptop GPU (6,000 steps, batch 128, Apple M-series). It is a starting point that shows
what a submission must get right, not a strong baseline. Directions that should help include better encoders
(ViTs, patch-level latents), longer training, a transformer predictor over the history, stop-gradient or EMA
targets, and planning-aware objectives.

## Validate submission

```bash
uv run test_submission.py path/to/model.onnx
```

It loads and runs the model with the evaluator's own code and explains how to fix any problem it finds. Only
submit once validation passes. [example_submission/model.onnx](example_submission/model.onnx) is a model that
passes.

Once your model is ready, submit it at https://galilai-group.github.io/stable-challenge-world-models/
