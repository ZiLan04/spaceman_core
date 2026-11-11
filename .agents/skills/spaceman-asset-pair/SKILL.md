---
name: spaceman-asset-pair
description: Select, compose, inspect and maintain registered satellite–robot-arm combinations in spaceman_core through provider-neutral Python and JSON commands. Use for seeded random scenes, mount edits, satellite geometry replacement and rotary gripper inspection; does not launch training.
---

Operate from the repository root. This skill is ordinary Markdown plus Python/JSON, with no Codex, DeepSeek, API-key or model SDK dependency. Agents without skill discovery can read this file explicitly; a human can run exactly the same commands. Use [the full contract](references/operations.md) for edits, schemas, validation and future training boundaries.

## Select and inspect

For native agent function calling, register `scripts/assets/scene_tool.schema.json` and dispatch its JSON arguments to `scripts/assets/scene_tool.py` or `execute(request)`. Actions are `list_assets`, `plan_scene`, and `render_scene`; render calls are serial and require unique run IDs. For another agent or an OpenAI-compatible DeepSeek API, read `docs/asset-pairs/DEEPSEEK_DEMO.md`. Its optional live API adapter is not a deployment of a specific harness, and a Codex demo does not establish that DeepSeek has run.

Prepare dependencies with the repository's Genesis environment (numpy, scipy, PyYAML; psutil for bounded runs; pytest for checks). Materialize compressed large meshes once:

```bash
python scripts/assets/materialize_meshes.py
python src/asset_pair_api.py list
python src/asset_pair_api.py plan --carrier satellite_17 --arm airbus_vispa --target gps --seed 42 --output artifacts/plan.json
python src/asset_pair_api.py plan --seed 42 --output artifacts/random-plan.json
python -m pytest src/test/test_agent_asset_pair.py -q
python safe_run.py --seconds 240 --rss-gib 8 --log artifacts/smoke.log -- python src/test/asset_pair_agent_smoke.py --carrier satellite_17 --arm airbus_vispa --target gps --render
```

For a small reproducible serial random test:

```bash
python src/test/random_registered_pair_test.py --seed 42 --count 2 --render --output artifacts/random-42
```

Python callers add `src` to their import path. Call `make_scene_plan(...)`, initialize Genesis, then `build_scene_entities(scene, plan)`, `scene.build()`, `apply_initial_pose(entities, plan)`. Topology changes require a new scene/process, not live hot swapping. The API does not import Genesis until scene construction.

The current library has 27 satellite registrations and two arms (Starlink, VISPA), with 54 precomposed URDFs. `satellite_6` is incomplete and excluded by default: random selection has 26 eligible satellites. `catch01` has 12 unmeshed CAD faces. Both arms now have Kinova JACO articulated two-finger hands at 2× source geometry scale, approximately 0.217 m palm width; arm and satellite scales stay unchanged. Each has six arm and four revolute finger joints, no prismatic finger joint or mimic. Public command 1=open, 0=curl; proximal angles `1-command`, distal angles `.65*(1-command)` radians. Resolve all joint indices by name.

## Maintain models and mounts

Read the maintenance section in [operations.md](references/operations.md) before edits. Use an isolated copy, not another person's checkout. Mount xyz is metres; mount RPY radians, order XYZ with `Rz @ Ry @ Rx`. Spawn Euler is degrees; quaternion is wxyz. Do not confuse satellite mount pose with the per-pair adapter transform or independent target grasp sites.

```bash
# Dry-run; replace these example coordinates with reviewed coordinates.
python scripts/assets/maintain.py --satellite satellite_17 --xyz 0.21 1.27 2.35 --rpy 0 -1.46 1.97
# Same command with --apply writes backups and rebuilds both arm combinations.
```

The rebuild synchronizes mount JSON, URDF frame, composed URDFs, bounds, catalog and YAML calibration state. It invalidates previous checks; it does not automatically design a conformal support or prove a new mount is safe. After geometry replacement run the same command without xyz/rpy. For adapter edits, change registry per-pair `mount_to_arm_root_*` then rebuild that satellite. For arm-wide edits, rebuild every affected satellite and update arm controls/TCP deliberately; do not run the old historical export script, which restores the old gripper.

## Evidence and boundaries

Plans are separated inspection poses with fixed roots and collisions disabled, not a reachable capture scene. `task_ready=false` is deliberate; the API rejects task/training use. Grasp sites are body-surface candidates, not confirmed handles. TCP, control gains, scaled inertia and free-floating dynamics remain uncalibrated. Old eight-joint checkpoints cannot be reused unchanged with the new ten-joint state.

Use CPU, one scene at a time, through `safe_run.py`: 8 GiB child-tree RSS cap, 240 s limit, two threads, GPUs disabled, stop if host available RAM is below 16 GiB. It only stops its own child tree. The RSS watcher is polling, not a kernel hard memory limit; do not promise zero overshoot. Stop a batch after the first failure and inspect its log before retrying. Do not launch PPO/VLA or modify other users' code as part of inspection.

Report IDs, seed, revision, actual tests, artifact paths and validation scope. Original 2026-11-10 checks covered open/closed endpoints at zero arm pose; two CPU Genesis smoke runs verified loading, initialization and rendering. These are not contact-grasp, swept-motion or task-success tests. Historical reports do not apply after a mount/model revision. Future StarVLA integration is described as a plan only in `docs/asset-pairs/FUTURE_TRAINING.md`.
