# Motive (57) P6 calibration C3D files

This folder contains lightweight Motive (57) calibration trials generated from
the original P6 Motive capture folder.

The files were renamed with generic names so the GUI can auto-assign them:

- `Example_Static.c3d`: main static C3D
- `Example_LHip.c3d`, `Example_RHip.c3d`: hip SCoRE functional trials
- `Example_LKnee.c3d`, `Example_RKnee.c3d`: knee SARA functional trials
- `Example_LAnkle.c3d`, `Example_RAnkle.c3d`: ankle SCoRE functional trials

To keep the example small and non-identifying, capture skeleton prefixes are
stripped from marker labels, file names are generic, frames are filtered to
valid frames for the corresponding static/SCoRE/SARA role, one valid frame out
of five is preserved, and analog channels are removed.

The files can be regenerated with:

```bash
python examples/prepare_motive_57_p6_c3ds.py
```

The GUI can be launched directly on this template with:

```bash
PYTHONPATH=. MPLCONFIGDIR=/tmp/mplconfig python examples/launch_motive_57_p6_model_editor.py
```

The same data can be used from the model creation CLI with:

```bash
PYTHONPATH=. MPLCONFIGDIR=/tmp/mplconfig python -m biobuddy.gui.build_c3d_model --p6-motive
```
