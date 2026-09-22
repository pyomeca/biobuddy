# Full-body Model202 calibration C3D files

This folder contains lightweight Model202 calibration trials generated from
`/Users/mickaelbegon/Downloads/MODEL202`.

The files were renamed with generic names so the GUI can auto-assign them. The GUI uses the anatomical-posture
trial as the main C3D for this example:

- `Test_anato.c3d`: main anatomical-posture C3D from `Anato.c3d`
- `Test_main.c3d`: original main full-body marker C3D kept for comparison
- `Test_func_<child>_<parent>.c3d`: functional SCoRE/SARA trial for the
  corresponding child-parent segment pair from the original Model202 functional trials

To keep the example small, frames are first filtered to keep only frames where
all markers required by the corresponding static/SCoRE/SARA role are valid.
Then one valid frame out of ten is preserved, analog channels are removed, and
participant prefixes are stripped from marker labels.

The files can be regenerated with:

```bash
python examples/prepare_full_body_model202_c3ds.py
```
