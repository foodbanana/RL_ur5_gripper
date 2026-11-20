"""Turn a distilled student into a PPO checkpoint for RL fine-tuning: student -> actor, teacher's critic -> critic.

The distillation checkpoint holds the student (input: the ``student`` group) and the teacher actor; the PPO
checkpoint the teacher was trained in holds its critic (input: ``policy`` + ``critic``, unchanged in the student
task). No optimizer state: train with ``--no-resume_optimizer_state`` (or the runner's default when it is missing).

Usage (from the repository root):

    uv run python scripts/student_to_ppo.py --student logs/rsl_rl/drone_grasp_ur5_student/<run>/model_<n>.pt \\
        --teacher logs/rsl_rl/drone_grasp_ur5/<run>/model_<n>.pt --out <student run>/model_<n>_ppo.pt
    # then: isaaclab train --task DroneGraspRl-Grasp-UR5-Student --agent rsl_rl_student_ppo_cfg_entry_point \\
    #     --checkpoint <out> --no-resume_optimizer_state
"""

import argparse

import torch

parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.add_argument("--student", required=True)
parser.add_argument("--teacher", required=True)
parser.add_argument("--out", required=True)
parser.add_argument("--std", type=float, default=None, help="Set the actor's action noise std (default: keep).")
args = parser.parse_args()

student = torch.load(args.student, map_location="cpu", weights_only=False)
teacher = torch.load(args.teacher, map_location="cpu", weights_only=False)
actor = dict(student["student_state_dict"])
if args.std is not None:
    actor["distribution.std_param"] = torch.full_like(actor["distribution.std_param"], args.std)
print(f"[student_to_ppo] actor std {actor['distribution.std_param'].tolist()}")
torch.save(
    {"actor_state_dict": actor, "critic_state_dict": teacher["critic_state_dict"], "iter": 0, "infos": None}, args.out
)
print(f"[student_to_ppo] wrote {args.out}")
