# Rotary two-finger replacement — 2026-11-10

Both Starlink and VISPA now use the Kinova JACO two-finger articulated rotary hand, sourced from the official Kinovarobotics/kinova-ros repository (pinned commit and BSD-3-Clause license are in each arm's angular_source/). This is an industrial hand adapted as simulation geometry, not flight-qualified hardware.

Uniform hand/mesh/origin scale: 2.0. Palm width is approximately 0.217 m, chosen to match VISPA's ~0.217 m wrist diameter and Starlink's ~0.227 m original gripper-body width. These dimensions justify a visual simulation fit only, not flange bolts, payload or structural compatibility. Arm and satellite geometry scales are unchanged.

Each arm retains six revolute arm joints. The new hand has four revolute joints: two fingers, each with proximal and distal hinge. Total movable joints: 10. There are no prismatic finger joints and no mimic. A single synchronized curl command drives all four: internal u=0 open, u=1 curl; proximal q=u rad, distal q=.65*u rad. Public environment command retains 1=open, 0=close. The configured gains, torques and coupling are simulation placeholders, not a validated real controller.

The nominal curl range is deliberately below the upstream joint limits (1.51 and 2 rad); it is not an aperture specification or a force/contact controller. TCP is an approximate fixed approach reference and needs contact calibration. Finger-tip contact links are angular_tip_1 and angular_tip_2; controllers needing full-finger contacts must explicitly include proximal links as well.

The versioned runtime contains 27 satellites and 54 combinations, generated YAML, API and tests. Previous source library and server runtime remain intact. Revised registry collision reports apply only to zero arm posture and hand open/closed endpoints. They do not establish swept motion, containment, grasp success, structural strength or dynamics. Historical reports elsewhere refer to previous grippers; use docs/asset-pairs/angular_validation_20261110.json as historical evidence and current registry statuses for revisions.

Local arm ZIPs include old meshes as unused historical resources; arm.urdf references only the new hand. Import the entire extracted folder into the URDF viewer. Pose preview PNGs demonstrate open/half/curl geometry, not a physics grasp.

Any downstream training/checkpoint expecting the old 8-joint state requires explicit adaptation. No StarVLA training or automatic learning integration is performed here. Provider-neutral operation and maintenance instructions are in .agents/skills/spaceman-asset-pair/SKILL.md.
