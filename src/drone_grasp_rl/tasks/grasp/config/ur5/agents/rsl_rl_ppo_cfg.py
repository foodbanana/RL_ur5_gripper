# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlMLPModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg


@configclass
class PPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """Starting point for the UR5 drone grasp; tuned once the grasp rewards are in."""

    num_steps_per_env = 24
    max_iterations = 1500
    save_interval = 100
    experiment_name = "drone_grasp_ur5"
    # The actor sees only what the real robot measures; the critic also gets simulator-only state.
    obs_groups = {"actor": ["policy"], "critic": ["policy", "critic"]}
    actor = RslRlMLPModelCfg(
        hidden_dims=[256, 128, 64],
        activation="elu",
        obs_normalization=True,
        # 0.5: with 1.0 the arm steps were too noisy to slide the open fingers past the drone
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=0.5),
    )
    critic = RslRlMLPModelCfg(
        hidden_dims=[256, 128, 64],
        activation="elu",
        obs_normalization=True,
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        # 0.006 let the action std grow (0.84 -> 1.13) while the approach stalled; 0.001 let it collapse to 0.05
        # before the gripper ever closed near the drone
        entropy_coef=0.003,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-4,
        schedule="adaptive",
        # 0.995 at 30 Hz = about 6.7 s horizon. With 0.98 (1.7 s) the success paid ~4.5 s after the close command
        # was discounted to ~6 %, and the policy farmed the per-step proximity rewards instead of grasping.
        gamma=0.995,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class StudentPPORunnerCfg(PPORunnerCfg):
    """RL fine-tuning of the distilled student: the actor sees only the deployable ``student`` group (ArUco marker
    drone pose, joints, last actions, 3 steps); the critic keeps the teacher's privileged input. Start from
    ``scripts/student_to_ppo.py`` (student weights as the actor, the teacher's critic)."""

    experiment_name = "drone_grasp_ur5_student"
    obs_groups = {"actor": ["student"], "critic": ["policy", "critic"]}

    def __post_init__(self):
        super().__post_init__()
        self.algorithm.entropy_coef = 0.0005
        self.algorithm.learning_rate = 3.0e-5
