# Husky A200 (moving base, visual)

`husky.usda` + `payloads/`: the Clearpath Husky A200 description converted with the Isaac Sim URDF importer, copied
from the data collection repo (`isaacsim/assets/bases/husky`). Do not edit by hand.

- Source: https://github.com/husky/husky, branch `humble-devel`, commit `729f8aa45ccd86fa33a05e07ef698c52c451cd9c`,
  package `husky_description`, xacro defaults (chassis, top plate, rails, bumpers, 4 wheels; no sensors or arch)
- License: BSD-3-Clause (Clearpath Robotics)
- The task loads the `none` physics variant (visual only): as in the data collection repo the Husky moves kinematically
  along a set trajectory (no wheel physics). A kinematic mount body carries it and the robot (`assets/bases.py`).
- Dimensions (base_link frame, m): chassis 46.0 kg, wheels radius 0.1651 at (+-0.256, +-0.2854, 0.0328), base_link
  0.132 above the floor, top plate link at (0.0812, 0, 0.245). UR5 mount from Clearpath's Husky + UR5 description
  (`indigo-devel` `ur5_arm_base`): (0.3058, 0, 0.0065) on the top plate link, yaw -90 deg (provisional, the real
  Husky's configuration is not measured yet).
