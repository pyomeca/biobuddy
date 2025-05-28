"""
This example shows how to read and write scaling configurations.
"""

import os
import logging
from pathlib import Path

from biobuddy import (
    ScaleTool,
    BiomechanicalModelReal,
)

_logger = logging.getLogger(__name__)


def main():
    # Configure logging
    logging.basicConfig(
        level=logging.DEBUG,  # Set the logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            # logging.FileHandler("app.log"),  # Log to a file
            logging.StreamHandler()  # Log to the console
        ],
    )

    # Paths
    current_path_file = Path(__file__).parent
    biomod_config_filepath = f"{current_path_file}/models/arm26_allbiceps_1dof.bioMod"
    biomod_config_filepath_new = f"{current_path_file}/models/arm26_allbiceps_1dof_new.bioMod"
    osim_config_filepath = f"{current_path_file}/models/arm26_allbiceps_1dof.xml"

    # --- Reading a .bioMod scaling configuration and translating it into a .xml configuration --- #
    # Read an .bioMod file
    original_model = BiomechanicalModelReal().from_biomod(filepath=biomod_config_filepath)

    scaling_configuration = ScaleTool(original_model).from_biomod(
        filepath=biomod_config_filepath,
    )

    # And convert it to a .xml file
    scaling_configuration.to_xml(osim_config_filepath)

    # Read the .xml file back
    new_xml_scaling_configuration = ScaleTool(original_model).from_xml(filepath=osim_config_filepath)

    # Rewrite it into a .bioMod to compare with the original one
    new_xml_scaling_configuration.to_biomod(biomod_config_filepath_new, append=False)

    os.remove(osim_config_filepath)
    os.remove(biomod_config_filepath_new)


if __name__ == "__main__":
    main()
