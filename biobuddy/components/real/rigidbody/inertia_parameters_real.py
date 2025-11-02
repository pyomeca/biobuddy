from typing import Callable

from lxml import etree
import numpy as np

from .protocols import CoordinateSystemRealProtocol
from ..biomechanical_model_real import BiomechanicalModelReal
from ....utils.aliases import Points, points_to_array, inertia_to_array
from ....utils.protocols import Data
from ....utils.linear_algebra import RotoTransMatrix


class InertiaParametersReal:
    def __init__(
        self,
        mass: float = None,
        center_of_mass: Points = None,
        inertia: Points = None,
    ):
        """
        Parameters
        ----------
        mass
            The mass of the segment with respect to the full body
        center_of_mass
            The position of the center of mass from the segment coordinate system on the main axis
        inertia
            The inertia xx, yy and zz parameters of the segment
        """
        self.mass = mass
        self.center_of_mass = center_of_mass
        self.inertia = inertia

    @property
    def mass(self) -> float:
        return self._mass

    @mass.setter
    def mass(self, value: float):
        self._mass = value

    @property
    def center_of_mass(self) -> np.ndarray:
        return self._center_of_mass

    @center_of_mass.setter
    def center_of_mass(self, value: Points):
        self._center_of_mass = points_to_array(points=value, name="center of mass")

    @property
    def inertia(self) -> np.ndarray:
        return self._inertia

    @inertia.setter
    def inertia(self, value: Points):
        self._inertia = inertia_to_array(value)

    def to_biomod(self):
        # Define the print function, so it automatically formats things in the file properly
        if self.mass is not None:
            out_string = f"\tmass\t{self.mass}\n"

        if np.any(self.center_of_mass):
            com = np.nanmean(self.center_of_mass, axis=1)[:3]
            out_string += f"\tCenterOfMass\t{com[0]:0.6f}\t{com[1]:0.6f}\t{com[2]:0.6f}\n"

        if np.any(self.inertia):
            out_string += f"\tinertia\n"
            out_string += f"\t\t{self.inertia[0, 0]:0.6f}\t{self.inertia[0, 1]:0.6f}\t{self.inertia[0, 2]:0.6f}\n"
            out_string += f"\t\t{self.inertia[1, 0]:0.6f}\t{self.inertia[1, 1]:0.6f}\t{self.inertia[1, 2]:0.6f}\n"
            out_string += f"\t\t{self.inertia[2, 0]:0.6f}\t{self.inertia[2, 1]:0.6f}\t{self.inertia[2, 2]:0.6f}\n"

        return out_string

    def to_urdf(self, link: etree.Element):

        inertial = etree.SubElement(link, "inertial")

        if self.mass is not None:
            mass_elt = etree.SubElement(inertial, "mass")
            mass_elt.set("value", str(self.mass))

        if np.any(self.center_of_mass):
            com = np.nanmean(self.center_of_mass, axis=1)[:3]
            com_elt = etree.SubElement(inertial, "origin")
            com_elt.set("rpy", f"0 0 0")
            com_elt.set("xyz", f"{com[0]:0.6f} {com[1]:0.6f} {com[2]:0.6f}")

        if np.any(self.inertia):
            inertia_elt = etree.SubElement(inertial, "inertia")
            inertia_elt.set("ixx", str(self.inertia[0, 0]))
            inertia_elt.set("ixy", str(self.inertia[0, 1]))
            inertia_elt.set("ixz", str(self.inertia[0, 2]))
            inertia_elt.set("iyy", str(self.inertia[1, 1]))
            inertia_elt.set("iyz", str(self.inertia[1, 2]))
            inertia_elt.set("izz", str(self.inertia[2, 2]))

    def to_osim(self):
        """
        Generate OpenSim XML representation of inertia parameters.
        Note: In OpenSim, inertia parameters are written directly in the Body element,
        so this method returns a dictionary for use by the body writer.
        """
        # OpenSim handles inertia at the body level, not as a separate element
        # This method returns the data in a format suitable for the body writer
        inertia_dict = {}

        if self.mass is not None:
            inertia_dict["mass"] = self.mass

        if np.any(self.center_of_mass):
            com = np.nanmean(self.center_of_mass, axis=1)[:3]
            inertia_dict["mass_center"] = com

        if np.any(self.inertia):
            inertia_dict["inertia"] = self.inertia

        return inertia_dict
