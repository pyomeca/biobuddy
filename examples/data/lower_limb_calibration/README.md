# Lower-Limb Calibration C3D Example Files

These files are lightweight calibration trials for the C3D model-creation GUI examples.

They were generated from the C3D files in `/Users/mickaelbegon/Downloads/calibration_files` with `ezc3d`:

- one frame out of ten was kept;
- marker coordinates were rotated by `pi` around the global `Y` axis, so `(x, y, z)` became `(-x, y, -z)`;
- participant namespaces were removed from point labels, for example `P01_MH:LASI` became `LASI`;
- analog data were removed;
- point frame rate was divided by ten.

The lower-limb preset with functional trials expects these file names in the example dataset:

- `Test_func_anat.c3d`
- `Test_func_lhip.c3d`
- `Test_func_lknee.c3d`
- `Test_func_lankle.c3d`
- `Test_func_rhip.c3d`
- `Test_func_rknee.c3d`
- `Test_func_rankle.c3d`

The template matches participant-independent patterns such as `*func_lhip.c3d`, so another participant prefix can be
used without changing the model definition.

Additional calibration files are included for examples that need an anatomical posture or trunk functional trial:

- `anatomical_posture.c3d`
- `functional_trunk.c3d`
