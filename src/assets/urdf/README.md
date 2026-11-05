# URDF asset layout

The data-driven fleet assets use three canonical directories:

- `arms/`: standalone manipulator URDFs and their local meshes.
- `satellites/`: standalone satellite URDFs, local meshes, and mount metadata.
- `combinations/`: one merged URDF for each satellite/manipulator pair. Mesh paths
  remain relative to `arms/` and `satellites/` so geometry is not duplicated.

Runtime registration lives in `src/configs/assets`. Each public asset has one YAML
file. Shared manipulator defaults are in `src/configs/assets/_profiles` and are
included with `extends`; files below `_profiles` are not registered directly.

Fields listed under `parameters.calibration.pending` are placeholders that require
task-specific measurements. In particular, the default grasp site is currently the
origin of each geometry-screened satellite mount frame. The URDF mount transforms,
link names, joint names, mesh paths, joint limits, and known SLDASM9 control values
come from the supplied assets or the existing project configuration.

`airbus_vispa` has no gripper joints in its source URDF. Its standalone and combined
configs therefore declare an empty gripper and `capture_capability:
no_gripper_defined`; they can be loaded and controlled as six-axis manipulators but
cannot complete the current two-finger capture task until an end effector is added.
