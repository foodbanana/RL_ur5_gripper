"""Extend a trained RSL-RL checkpoint to a larger observation, keeping its behavior.

The new observation entries are inserted into the first layer of the actor and the critic with zero weights and
into the observation normalizers with mean 0 / std 1, so the extended networks compute exactly what the old ones
did until training starts using the new entries. Used to turn the deployable 30-entry grasp policy into the
privileged teacher (52 entries: the 30 old ones first, 22 new ones after).

The critic's input is the policy group followed by the critic group, so its new entries go right after the old
policy part. The Adam moments of both first layers are extended the same way (zeros for the new columns) and the
optimizer state is kept: fine-tuning a converged policy with a fresh Adam moved every weight by about the learning
rate in the first updates, and the policy stopped approaching the drone within 10 iterations.

Usage (from the repository root):

    uv run python scripts/expand_policy_inputs.py --checkpoint logs/.../model_12691.pt \\
        --out logs/.../model_12691_teacher.pt --old_policy_dim 30 --new_policy_dim 52 --std 0.3
    # then train with --checkpoint <out> (the optimizer state is kept)
"""

import argparse

import torch

parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--out", required=True)
parser.add_argument("--old_policy_dim", type=int, required=True)
parser.add_argument("--new_policy_dim", type=int, required=True)
parser.add_argument("--std", type=float, default=None, help="Reset the action noise std to this value.")
args = parser.parse_args()


def insert_inputs(state: dict, prefix: str, at: int, count: int):
    """Insert ``count`` zero-weight inputs at column ``at`` of ``prefix``'s first layer and normalizer."""
    weight = state[f"{prefix}mlp.0.weight"]
    zeros = torch.zeros(weight.shape[0], count, dtype=weight.dtype)
    state[f"{prefix}mlp.0.weight"] = torch.cat([weight[:, :at], zeros, weight[:, at:]], dim=1)
    for key, fill in (("_mean", 0.0), ("_var", 1.0), ("_std", 1.0)):
        name = f"{prefix}obs_normalizer.{key}"
        if name in state:
            old = state[name]
            new = torch.full((old.shape[0], count), fill, dtype=old.dtype)
            state[name] = torch.cat([old[:, :at], new, old[:, at:]], dim=1)
    print(f"[expand] {prefix}first layer {tuple(weight.shape)} -> {tuple(state[f'{prefix}mlp.0.weight'].shape)}")


def main():
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    count = args.new_policy_dim - args.old_policy_dim
    actor, critic = ckpt["actor_state_dict"], ckpt["critic_state_dict"]
    assert actor["mlp.0.weight"].shape[1] == args.old_policy_dim, "checkpoint actor input does not match"
    insert_inputs(actor, "", args.old_policy_dim, count)
    insert_inputs(critic, "", args.old_policy_dim, count)
    if args.std is not None:
        old_std = actor["distribution.std_param"].clone()
        actor["distribution.std_param"].fill_(args.std)
        print(f"[expand] action std {old_std.mean():.3f} -> {args.std}")
    optim = ckpt.get("optimizer_state_dict")
    if optim is not None:
        old_actor = (actor["mlp.0.weight"].shape[0], args.old_policy_dim)
        old_critic = (critic["mlp.0.weight"].shape[0], critic["mlp.0.weight"].shape[1] - count)
        for index, state in optim["state"].items():
            for key in ("exp_avg", "exp_avg_sq"):
                tensor = state.get(key)
                if tensor is not None and tuple(tensor.shape) in (old_actor, old_critic):
                    zeros = torch.zeros(tensor.shape[0], count, dtype=tensor.dtype)
                    at = args.old_policy_dim
                    state[key] = torch.cat([tensor[:, :at], zeros, tensor[:, at:]], dim=1)
                    if key == "exp_avg":
                        print(f"[expand] optimizer state {index}: {tuple(tensor.shape)} -> {tuple(state[key].shape)}")
    torch.save(ckpt, args.out)
    print(f"[expand] wrote {args.out}")


if __name__ == "__main__":
    main()
