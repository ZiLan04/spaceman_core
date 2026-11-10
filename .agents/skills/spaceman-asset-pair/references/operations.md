# Asset pair operations and edit contract

## Files and ownership

| Source | Meaning | Derived outputs |
|---|---|---|
| `src/assets/registered_library/satellites/<id>/satellite.urdf` plus local mesh files | Satellite body and attached support geometry | Combination URDFs and bounds |
| `satellites/<id>/mount_config.json` | `mount_frame`, `mount_xyz_m`, `mount_rpy_rad`, source/scale notes | Mount joint in satellite URDF and catalog |
| `registered_library/registry.json` | Satellite/arm paths, root links, each pair's mount-to-arm transform and assembly path | Composed URDFs |
| `registered_library/arms/<id>/arm.urdf` | Arm, adapter, four rotary finger joints and nominal TCP | All combinations using the arm |
| `src/configs/assets/<id>.yaml`, `<satellite>__<arm>.yaml` | Upstream registry contract, actuator control, spawn defaults, grasp references | Typed specs consumed by Genesis |
| `registered_library/catalog.json` | Agent-facing catalog, initial joint name map, eligibility, target grasp sites and bounds | API scene plans |

The existing schema predates this maintenance tool and is not a fully normalized single-source database. The tool explicitly synchronizes mount pose and dependent geometry metadata. YAML control profiles, catalog initial-joint maps, TCP and grasp sites still require deliberate coordinated edits; do not imply that changing one file updates everything.

Transform contract: `T_satellite_arm_root = T_satellite_mount @ T_mount_arm_root`. Mount configuration describes the mount frame in its URDF parent link. Existing assets use the body root parent. Per-pair transforms describe how each arm sits on the common mount. A model with a newly introduced non-root parent must be explicitly adapted before using existing body-space catalog/grasp conventions.

## Calling without any agent

```python
import sys
sys.path.insert(0, 'src')
from asset_pair_api import list_assets, make_scene_plan
ids = list_assets()
plan = make_scene_plan('satellite_17', 'airbus_vispa', 'gps', seed=42)
# Or let the seeded selector choose all three:
random_plan = make_scene_plan(seed=42)
```

`list` and `plan` print JSON to stdout. `--output` also persists the plan. Exit code is nonzero on invalid IDs, partial assets without opt-in, overlapping spawn bounds, nonzero unvalidated initial joints, or training use. Plans contain world-space candidate target positions/normals/orientations computed from target pose. Merely editing a saved plan does not bypass validation: scene construction re-resolves canonical specs and initial joints.

A CLI plan is a description, not a running scene. Smoke tests choose IDs via their own arguments. `registered_asset_pair_viewer_test.py` is the upstream interactive viewer; prefer the bounded headless smoke command for server checks.

## Changing mount placement

1. Create a separate branch/copy and materialize meshes. Inspect the actual body geometry. Reserve docking ends; choose a body patch near the intended working direction and within arm reach. Do not mount on solar panels, antenna tips or unsupported appendages.
2. Run `scripts/assets/maintain.py --satellite <id>` for a dry-run. Pass reviewed `--xyz x y z --rpy roll pitch yaw`, then `--apply` to synchronize/recompose. JSON functions are also importable as `rebuild(root, satellite, xyz, rpy, apply)` after adding `scripts/assets` to `sys.path`.
3. Backups are written under `.asset_backups/<timestamp>/` before derived edits. Failed rebuilds restore the files owned by the tool. Backups reflect the files as found at invocation; manually edited meshes/registry before invocation need their own Git snapshot. Rebuild is transactional for its listed files, not arbitrary prior user changes.
4. The existing support is attached to the mount frame and follows it. Its old conformal shape may no longer fit the new surface. Refit/replace `mount_support.stl` or corresponding geometry; the tool invalidates the old surface-fit claim rather than claiming a new fit.
5. A per-pair root offset/orientation is in registry `mount_to_arm_root_xyz_m` / `mount_to_arm_root_rpy_rad`; edit those independently when the two arms need different bases, then rebuild the satellite. Rebuild updates the catalog with these values.
6. Recheck actual support contact, body intersections, finger/arm self-collision, open/intermediate/closed poses, task reach, and swept motion as applicable. The included smoke tests cover loading only. Do not set `task_ready=true` from a rendered image.

The tool recomputes zero-joint visual bounds (binary STL and boxes) in chunks. New unsupported mesh formats or visual primitives fail explicitly; convert them or extend bounds support first. It does not manufacture structural validation. Each rebuilt pair gets `edited_requires_validation`; old check fields are removed. Archived `docs/asset-pairs/*20261110*` remain historical evidence only.

## Replacing a satellite model

For a compatible replacement, keep the registered ID, base/root and mount/support link names stable. Put portable local mesh paths in `satellite.urdf`, update source provenance, scaling and mount config, then rebuild that ID. If the physical body size changes, reevaluate mount placement, candidate grasp sites and all related camera/task poses. Do not silently reuse old grasp coordinates.

For a new ID or incompatible topology, add the satellite entry to registry and catalog, provide its standalone YAML, create one pair entry and paired YAML per arm, then compose/rebuild. Start from a nearby asset type but assign unique IDs and correct root references. Explicitly mark partial source assemblies ineligible. The current tool rebuilds registered IDs; it does not automatically infer mounts for arbitrary CAD or register a brand-new ID.

Changing arm geometry or adding another hand requires updating arm URDF, root reference, joint names/limits/control arrays, TCP/contact links, arm catalog and all paired YAML. Rebuild every satellite using that arm. The current angular hand has joints `angular_joint_1`, `angular_joint_tip_1`, `angular_joint_2`, `angular_joint_tip_2`; tip contact links `angular_tip_1/2`. Inspect source provenance/license in each arm's `angular_source/`. Scale 2× is a simulation dimensional fit, not real Kinova hardware or flight qualification.

## Initial pose and grasp-site edits

Scene-level `carrier_position`, `target_position` and Euler-degree orientation overrides are arguments to `make_scene_plan`. These are independent of mount transforms. Current API intentionally allows only the screened zero arm preset; adding nonzero initial presets requires code/validation changes rather than quietly disabling the guard.

Grasp candidates are independent from mount sites: edit catalog `satellites[id].grasp_sites` and standalone/pair YAML `grasp_sites` together. Positions/normals are in body coordinates, quaternion wxyz. Update status and provenance. The maintenance tool preserves them and never re-labels a moved mount as a verified grasp point. If the body is replaced, explicitly reevaluate these entries.

## Validation and delivery

Run `python -m pytest src/test/test_agent_asset_pair.py src/test/test_asset_registry.py src/test/test_scene_registry.py -q` when changing contracts. Run serial bounded CPU smoke scenes for both arms; preserve stdout logs, resource summaries, JSON plans and PNGs. For geometry revisions add the appropriate surface/reach checks; no built-in test certifies capture. Successful inspection, contact grasp and policy success rate are three separate evidence levels.

Before committing, exclude simulation artifacts, training outputs, credentials and caches. Large source STL meshes over 50 MB are stored as `.stl.gz` plus checksum manifest to remain below GitHub's 100 MB single-blob limit. `materialize_meshes.py` restores exact bytes, verifies SHA256 and will not overwrite a different existing file. Run it on every fresh checkout. If source geometry changes, regenerate the compressed copy and manifest before committing.

Use the maintained runtime in this repository. Historical local tools such as `build_registered_bundle.py` and `replace_angular_grippers.py` were one-time migrations tied to a Windows workspace; they are not this skill's portable rebuild entrypoint.
