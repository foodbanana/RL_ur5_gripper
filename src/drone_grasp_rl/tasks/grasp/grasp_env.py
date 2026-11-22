# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""The grasp environment: Isaac Lab's manager-based RL environment plus the Newton robot-to-base collision filter.

On a moving base the robot stays a separate asset (other arms can be mounted). PhysX filters the robot base bodies
against the base body with ``UsdPhysics.FilteredPairsAPI`` (``mdp.attach_robot_to_base``). Newton's importer reads
those pairs per asset, and Isaac Lab 3.0 builds every asset separately, so a pair between two assets is lost.
Newton instead gets the same pairs from a world-builder hook (``isaaclab_newton.cloner.newton_builder_world_hook``)
that runs while the scene is cloned, when both assets of a world are in one builder.
"""

from __future__ import annotations

import contextlib

from isaaclab.envs import ManagerBasedRLEnv


def _mount_filter_hook(robot_bodies: list[str], base_body: str):
    """World-builder hook adding collision filter pairs between the robot bodies and the base body.

    Bodies are matched by the end of their label (``/<path relative to the environment>``) in the first world; the
    other worlds are copies of the same prototype, so the bodies are at the same offsets from the end of the builder.
    """
    offsets: dict[str, list[int]] = {}

    def hook(builder, world, pos, quat):
        if not offsets:
            current = [i for i in range(builder.body_count) if builder.body_world[i] == builder.current_world]

            def find(path):
                return [builder.body_count - i for i in current if builder.body_label[i].endswith("/" + path)]

            offsets["robot"] = [o for path in robot_bodies for o in find(path)]
            offsets["base"] = find(base_body)
            if len(offsets["robot"]) != len(robot_bodies) or len(offsets["base"]) != 1:
                raise RuntimeError(
                    f"Mount collision filter: bodies not found (robot {robot_bodies}, base {base_body})."
                )
        robot = [builder.body_count - o for o in offsets["robot"]]
        base = builder.body_count - offsets["base"][0]
        for body in robot:
            for shape_a in builder.body_shapes[body]:
                for shape_b in builder.body_shapes[base]:
                    builder.add_shape_collision_filter_pair(shape_a, shape_b)

    return hook


class GraspRLEnv(ManagerBasedRLEnv):
    """:class:`ManagerBasedRLEnv` that also filters robot-to-base collisions on Newton (see the module docstring).

    Active when the physics is Newton and ``actions.robot_mount`` is a :class:`mdp.SoftMountCfg` with
    ``filtered_robot_bodies``; otherwise it is the plain manager-based environment.
    """

    def __init__(self, cfg, render_mode: str | None = None, **kwargs):
        from isaaclab_newton.physics import NewtonCfg

        from .mdp import SoftMountCfg

        mount = getattr(cfg.actions, "robot_mount", None)
        with contextlib.ExitStack() as stack:
            if (
                isinstance(cfg.sim.physics, NewtonCfg)
                and isinstance(mount, SoftMountCfg)
                and mount.filtered_robot_bodies
            ):
                from isaaclab_newton.cloner import newton_builder_world_hook

                hook = _mount_filter_hook(list(mount.filtered_robot_bodies), mount.base_body_path)
                stack.enter_context(newton_builder_world_hook(hook))
            super().__init__(cfg, render_mode, **kwargs)
