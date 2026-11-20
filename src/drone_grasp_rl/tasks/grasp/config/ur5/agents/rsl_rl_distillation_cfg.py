# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlDistillationAlgorithmCfg, RslRlDistillationRunnerCfg, RslRlMLPModelCfg


@configclass
class DistillationRunnerCfg(RslRlDistillationRunnerCfg):
    """Teacher (PPO actor on the ``policy`` group) -> student on the deployable ``student`` group (DAgger-style: the
    student acts, the teacher labels every visited state). Load the teacher with ``--checkpoint`` <PPO model>."""

    num_steps_per_env = 24
    max_iterations = 1500
    save_interval = 100
    experiment_name = "drone_grasp_ur5_student"
    obs_groups = {"student": ["student"], "teacher": ["policy"]}
    student = RslRlMLPModelCfg(
        hidden_dims=[256, 128, 64],
        activation="elu",
        obs_normalization=True,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=0.05),
    )
    teacher = RslRlMLPModelCfg(
        hidden_dims=[256, 128, 64],
        activation="elu",
        obs_normalization=True,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=0.0),
    )
    algorithm = RslRlDistillationAlgorithmCfg(
        num_learning_epochs=2,
        learning_rate=1.0e-4,
        gradient_length=1,
        max_grad_norm=1.0,
        # the teacher's raw actions reach |12| (the action terms clip to [-1, 1]); with MSE and 5e-4 the student
        # diverged within 3 iterations (loss 5.7 -> 57685)
        loss_type="huber",
    )
