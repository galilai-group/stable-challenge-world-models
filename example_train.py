"""Worked example: train a small latent world model on the corpus and export it.

    unzip train.zip -d corpus
    python example_train.py --corpus corpus --out model.onnx

The model is a JEPA-style world model with no pixel decoder:

    encoder    the HIST context frames, stacked on channels -> a small CNN -> z (D=256)
    predictor  z_{k+1} = z_k + MLP([z_k, a_k]), where a_k are the next SKIP actions; rolled out HORIZON steps

Training slides a window over each episode: the encoder embeds the context window and the HORIZON windows after
it; the predictor rolls out from the first embedding, and its predictions are pulled toward the later embeddings
(mean squared error) and must pick out their own target among the batch (InfoNCE), while VCReg (stable-pretraining's
variance-covariance loss) keeps the embeddings from collapsing. Episodes are sampled uniformly, whatever their
length. No labels and no pixel reconstruction are used.

This is the organizers' reference model: a score of 1.0 on each task means matching it. It is a starting point,
not a strong baseline. It shows what every submission has to get right: the two-input contract, an encoder whose
output does not depend on the actions, a causal predictor, and a clean ONNX export.

    python example_train.py --floor --out floor.onnx   # the untrained, no-dynamics model that defines a score of 0
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import lightning as pl
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import stable_pretraining as spt
from stable_pretraining.backbone import MLP
from stable_pretraining.losses.utils import VCRegLoss

sys.path.insert(0, str(Path(__file__).parent))
import corpus_loader  # noqa: E402
from evaluate import ACTION_DIM, BATCH, HIST, HORIZON, SIZE, SKIP  # noqa: E402

OPSET = 17
SPAN_BACK, SPAN_FWD = (HIST - 1) * SKIP, HORIZON * SKIP


class Encoder(nn.Module):
    """[B, HIST, 3, 64, 64] in [0, 1] -> [B, dim]. GroupNorm, so each sample is encoded independently."""

    def __init__(self, dim=256, width=32):
        super().__init__()
        chans = [HIST * 3, width, 2 * width, 4 * width, 4 * width]
        layers = []
        for cin, cout in zip(chans[:-1], chans[1:]):  # 64 -> 32 -> 16 -> 8 -> 4
            layers += [nn.Conv2d(cin, cout, 4, stride=2, padding=1), nn.GroupNorm(8, cout), nn.GELU()]
        self.conv = nn.Sequential(*layers)
        self.proj = nn.Linear(chans[-1] * (SIZE // 16) ** 2, dim)

    def forward(self, frames):
        b = frames.shape[0]
        x = frames.reshape(b, HIST * 3, SIZE, SIZE) - 0.5
        return self.proj(self.conv(x).flatten(1))


class Predictor(nn.Module):
    """Residual one-step dynamics in latent space: z, a -> z + MLP([z, a])."""

    def __init__(self, dim=256, hidden=512):
        super().__init__()
        self.mlp = MLP(dim + ACTION_DIM * SKIP, [hidden, hidden, dim], activation_layer=nn.GELU)

    def forward(self, z, a):
        return z + self.mlp(torch.cat([z, a], -1))


class WorldModel(nn.Module):
    """The submission contract: (frames [B, HIST, 3, 64, 64], actions [B, HORIZON, 2*SKIP]) -> z [B, HORIZON+1, D]."""

    def __init__(self, dim=256, identity_dynamics=False):
        super().__init__()
        self.encoder = Encoder(dim)
        self.predictor = Predictor(dim)
        self.identity_dynamics = identity_dynamics  # the score-0 floor: z never changes

    def rollout(self, z, actions):
        out = [z]
        for k in range(actions.shape[1]):
            # (the floor keeps a zero-weight use of the actions so the export still has the `actions` input)
            z = z + 0 * actions[:, k, :1] if self.identity_dynamics else self.predictor(z, actions[:, k])
            out.append(z)
        return torch.stack(out, 1)

    def forward(self, frames, actions):
        return self.rollout(self.encoder(frames), actions)


class Windows(torch.utils.data.Dataset):
    """Every (episode, t) with HIST context frames before t and HORIZON steps after it.

    Returns frames uint8 [HIST + HORIZON, 64, 64, 3] at t - SPAN_BACK, ..., t + SPAN_FWD (SKIP apart) and the
    SPAN_FWD actions from t, grouped SKIP per step: [HORIZON, 2*SKIP].
    """

    def __init__(self, root):
        self.c = corpus_loader.load(root)
        starts, ts = [], []
        for i, n in enumerate(self.c.lengths):
            t = np.arange(SPAN_BACK, n - SPAN_FWD)
            starts.append(np.full(len(t), self.c.offsets[i]))
            ts.append(t)
        self.index = np.stack([np.concatenate(starts), np.concatenate(ts)], 1)
        # weight windows so every episode is drawn equally often, whatever its length
        self.weights = np.concatenate([np.full(len(t), 1.0 / len(t)) for t in ts])

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        s, t = self.index[i]
        rows = s + t + np.arange(-SPAN_BACK, SPAN_FWD + 1, SKIP)
        frames = torch.from_numpy(np.asarray(self.c.frames[rows]))
        actions = np.asarray(self.c.actions[s + t : s + t + SPAN_FWD]).reshape(HORIZON, ACTION_DIM * SKIP)
        return {"frames": frames, "actions": torch.from_numpy(actions.copy())}


def to_model_frames(u8):
    """uint8 [..., 64, 64, 3] -> float [..., 3, 64, 64] in [0, 1], as the evaluator feeds them."""
    return u8.float().div(255).movedim(-1, -3)


def info_nce(pred, target, temperature=0.1):
    logits = F.normalize(pred, dim=-1) @ F.normalize(target, dim=-1).T / temperature
    return F.cross_entropy(logits, torch.arange(len(pred), device=pred.device))


def jepa_forward(self, batch, stage):
    """Forward for ``spt.Module``: multi-step latent prediction + VCReg."""
    frames = to_model_frames(batch["frames"])  # [B, HIST + HORIZON, 3, 64, 64]
    b = frames.shape[0]
    windows = torch.stack([frames[:, k : k + HIST] for k in range(HORIZON + 1)], 1)
    target = self.backbone.encoder(windows.flatten(0, 1)).view(b, HORIZON + 1, -1)
    pred = self.backbone.rollout(target[:, 0], batch["actions"])[:, 1:]
    pred_loss = F.mse_loss(pred, target[:, 1:])
    # VCReg on the embeddings, and on their deviation from each clip's mean: the second term needs the content that
    # changes over time (where the agent is) to vary, which the first alone can satisfy by separating scenes.
    reg = self.vcreg(target.flatten(0, 1))
    reg_t = self.vcreg((target - target.mean(1, keepdim=True)).flatten(0, 1))
    # InfoNCE: each prediction must pick out its own target among the batch. Unlike the MSE it is scale-free, so the
    # model cannot satisfy it by shrinking the embeddings of a whole scene to one point.
    nce = sum(info_nce(pred[:, k], target[:, k + 1]) for k in range(HORIZON)) / HORIZON
    loss = 25 * pred_loss + reg + reg_t + nce
    self.log_dict({f"{stage}/pred": pred_loss, f"{stage}/vcreg": reg, f"{stage}/vcreg_t": reg_t,
                   f"{stage}/nce": nce}, prog_bar=True)
    return {"loss": loss}


def export(model, out: Path) -> None:
    """Export in eval mode with the contract's static shapes."""
    model.eval().to("cpu")
    frames = torch.rand(BATCH, HIST, 3, SIZE, SIZE)
    actions = torch.randn(BATCH, HORIZON, ACTION_DIM * SKIP)
    torch.onnx.export(model, (frames, actions), str(out), input_names=["frames", "actions"], output_names=["z"],
                      opset_version=OPSET, dynamo=False)
    print(f"wrote {out}", file=sys.stderr)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--corpus", type=Path)
    parser.add_argument("--out", type=Path, default=Path("model.onnx"))
    parser.add_argument("--steps", type=int, default=6000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--floor", action="store_true", help="export the untrained, no-dynamics floor model")
    args = parser.parse_args()
    torch.manual_seed(args.seed)

    if args.floor:
        export(WorldModel(identity_dynamics=True), args.out)
        return
    if args.corpus is None:
        parser.error("--corpus is required unless --floor")

    module = spt.Module(
        backbone=WorldModel(),
        vcreg=VCRegLoss(std_coeff=25, cov_coeff=1),
        forward=jepa_forward,
        optim={"optimizer": {"type": "AdamW", "lr": args.lr, "weight_decay": 1e-4},
               "scheduler": {"type": "CosineAnnealingLR", "T_max": args.steps}},
    )
    data = Windows(args.corpus)
    sampler = torch.utils.data.WeightedRandomSampler(data.weights, num_samples=args.steps * args.batch_size)
    loader = torch.utils.data.DataLoader(data, batch_size=args.batch_size, sampler=sampler,
                                         num_workers=args.workers, drop_last=True, persistent_workers=True)
    trainer = pl.Trainer(
        max_steps=args.steps,
        precision="bf16-mixed" if torch.cuda.is_available() else "32-true",
        logger=False,
        enable_checkpointing=False,
        log_every_n_steps=50,
    )
    spt.Manager(trainer=trainer, module=module, data=spt.data.DataModule(train=loader))()

    export(module.backbone, args.out)
    print(f"\nNow check it:  python test_submission.py {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
