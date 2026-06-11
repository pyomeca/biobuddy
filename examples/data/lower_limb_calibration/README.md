# Lower-Limb Calibration C3D Example Files

These files are lightweight calibration trials for the C3D model-creation GUI examples.

They were generated from the C3D files in `/Users/mickaelbegon/Downloads/calibration_files` with `ezc3d`:

- one frame out of ten was kept;
- marker coordinates were rotated by `pi` around the global `Y` axis, so `(x, y, z)` became `(-x, y, -z)`;
- participant namespaces were removed from point labels, for example `P01_MH:LASI` became `LASI`;
- analog data were removed;
- point frame rate was divided by ten.

The lower-limb preset with functional trials expects these generic file names:

- `main_markers.c3d`
- `functional_left_hip_score.c3d`
- `functional_left_knee_sara.c3d`
- `functional_left_ankle_score.c3d`
- `functional_right_hip_score.c3d`
- `functional_right_knee_sara.c3d`
- `functional_right_ankle_score.c3d`

Additional calibration files are included for examples that need an anatomical posture or trunk functional trial:

- `anatomical_posture.c3d`
- `functional_trunk.c3d`
