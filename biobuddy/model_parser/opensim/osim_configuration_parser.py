from enum import Enum
from time import strftime

import numpy as np
from xml.etree import ElementTree

from .utils import _is_element_empty


def _get_file_version(model: ElementTree) -> int:
    return int(model.getroot().attrib["Version"])


class OsimConfigurationParser:
    def __init__(
        self,
        filepath: str,
    ):
        """
        Reads and converts OpenSim configuration files (.xml) to a generic configuration.

        Parameters
        ----------
        filepath : str
            Path to the OpenSim configuration.xml file to read

        Raises
        ------
        RuntimeError
            If file version is too old or units are not meters/newtons
        """
        # Initial attributes
        self.filepath = filepath

        # Extended attributes
        self.configuration = ElementTree.parse(filepath)

        for element in self.configuration.getroot()[0]:
            if element.tag == "TODO":
                print("TODO")
            else:
                raise RuntimeError(
                    f"Element {element.tag} not recognize. Please verify your xml file or send an issue"
                    f" in the github repository."
                )

        self._read()

    def _read(self):
        """
        Read the configuration file
        """
        # Read the version
        self.version = _get_file_version(self.configuration)
