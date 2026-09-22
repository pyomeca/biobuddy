"""
This example shows how to launch the model editor GUI. The GUI allows to:
1) Edit an existing model by loading it with the "Open Model" button.
2) Create a new model using the experimental marker position from a static trial.
"""

from biobuddy import launch_model_editor


def launch_gui():
    """
    If you want to create a new model from your data, you can let your imagination run wild ;)
    Or if you want to follow a quick tutorial to see how the GUI works, you can:
    Tutorial 1:
    1. biobuddy/examples/data/lower_limb_calibration/
    2. biobuddy/examples/data/full_body_model202/
    """

    launch_model_editor()


if __name__ == "__main__":
    launch_gui()
