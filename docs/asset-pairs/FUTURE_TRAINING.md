# Deferred integration with RL / StarVLA

This branch implements provider-neutral model selection, scene inspection, maintainable mounts and rotary-hand composition. It does not implement an autonomous learning loop, transfer the previous StarVLA checkpoint, start PPO, or claim random-environment capture success.

Proposed next integration boundary (not an implemented interface):

1. Freeze an asset revision and export IDs, source hashes, actual geometry scales, mount/root transforms, joint limits, TCP/contact frames, cameras, candidate and validated task sites, task initialization distribution and calibration status.
2. Validate dynamics, collision/contact geometry, reachability and task success criteria. Replace inspection-only fixed/collision-disabled settings with reviewed task settings. Define variation ranges and held-out combinations/seeds.
3. Adapt the RL expert/action and observation contract to the selected arm and four-joint hand. Keep one policy-level gripper scalar only if coupling is explicitly defined; raw ten-joint state is not compatible with old eight-joint checkpoints by name alone.
4. Collect successful and informative failure rollouts under a bounded job runner. Record episode asset revision, random seed, units, images, state, actions and outcome. Define Delta-EEF and gripper semantics before LeRobot export.
5. Reuse the previously validated StarVLA pipeline only after the data/action/TCP contract matches. Fine-tune, evaluate closed loop on held-out seeds and combinations, stratify failures, and make one targeted intervention at a time.
6. A future agent orchestrator may call these deterministic stages and decide from evaluation evidence. Training requests need their own resource budget, monitoring, stopping and artifact contracts. Model metadata alone does not make learning autonomous or transferable to arbitrary satellites.

The current `make_scene_plan` rejects training purpose and always reports `task_ready=false`. Keep that guard until a separate reviewed task adapter exists. No training automation is installed by this branch.
