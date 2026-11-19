# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Curriculum terms for the drone grasp task."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

from isaaclab.managers import ManagerTermBase

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.managers import CurriculumTermCfg


class ramp_reward_weight(ManagerTermBase):
    """Move a reward weight linearly from ``initial_weight`` to ``final_weight`` over ``num_steps`` env steps,
    starting at ``start_step`` (``env.common_step_counter``, counted from the start of each training run).

    With ``min_success`` the ramp is performance based, like the command and terrain levels of unitree_rl_lab and
    Isaac Lab's ADR: it only advances (by the env steps that pass) while the judge's success rate over recently
    finished episodes is at least ``min_success``, and holds otherwise.

    New penalties are brought in this way so the critic and the policy adapt gradually, like Isaac Lab's contrib
    lift raises its action penalties (there with a single switch, :func:`isaaclab.envs.mdp.modify_reward_weight`).
    """

    def __init__(self, cfg: CurriculumTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._term_cfg = env.reward_manager.get_term_cfg(cfg.params["term_name"])
        self._progress_steps = 0
        self._last_step = 0
        self._success_rate = 0.0

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        env_ids: Sequence[int],
        term_name: str,
        initial_weight: float,
        final_weight: float,
        num_steps: int,
        start_step: int = 0,
        min_success: float | None = None,
        judge_term: str = "grasp_success",
    ) -> float:
        if min_success is None:
            progress = min(max((env.common_step_counter - start_step) / max(num_steps, 1), 0.0), 1.0)
        else:
            # success rate of the episodes finishing now (exponential average over resets)
            judge = env.termination_manager.get_term_cfg(judge_term).func
            phase = judge.phase[env_ids]  # env_ids is a slice on the first reset
            if phase.numel() > 0:
                # the curriculum runs at the start of the reset, before the judge resets: phase is the outcome
                held = (phase == 3).float().mean().item()
                self._success_rate = 0.95 * self._success_rate + 0.05 * held
            elapsed = env.common_step_counter - self._last_step
            self._last_step = env.common_step_counter
            if env.common_step_counter >= start_step and self._success_rate >= min_success:
                self._progress_steps += elapsed
            progress = min(self._progress_steps / max(num_steps, 1), 1.0)
        self._term_cfg.weight = initial_weight + (final_weight - initial_weight) * progress
        env.reward_manager.set_term_cfg(term_name, self._term_cfg)
        return self._term_cfg.weight
