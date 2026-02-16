import numpy as np
from lxml import etree
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

from .via_point_real import ViaPointReal
from ...ligament_utils import LigamentType, LIGAMENT_FUNCTION
from ...functions import InterpolationFunction


class LigamentReal:
    def __init__(
        self,
        name: str,
        ligament_type: LigamentType,
        origin_position: ViaPointReal,
        insertion_position: ViaPointReal,
        ligament_slack_length: float,
        stiffness: float = None,
        damping: float = None,
        force_length_function: InterpolationFunction = None,
        pcsa: float = None,
    ):
        """
        Parameters
        ----------
        name
            The name of the new contact
        ligament_type
            The type of the force
        origin_position
            The origin position of the force in the local reference frame of the origin segment
        insertion_position
            The insertion position of the force the local reference frame of the insertion segment
        ligament_slack_length
            The length of the ligament at rest
        stiffness
            The stiffness of the sping representing the ligament, if ligament_type is LigamentType.CONSTANT, LigamentType.LINEAR_SPRING, or LigamentType.SECOND_ORDER_SPRING
        damping
            The damping of the ligament, if ligament_type is LigamentType.CONSTANT, LigamentType.LINEAR_SPRING, or LigamentType.SECOND_ORDER_SPRING
        force_length_function
            The function giving the force-length relationship of the ligament, if ligament_type is LigamentType.FUNCTION
        pcsa
            The physiological cross-sectional area of the ligament, if ligament_type is LigamentType.FUNCTION
        """
        super().__init__()

        if ligament_type == LigamentType.FUNCTION:
            # You are using an OpenSim-like ligament
            if force_length_function is None:
                raise ValueError("The force-length function of the ligament must be provided for ligaments of type FUNCTION.")
            if pcsa is None:
                raise ValueError("The physiological cross-sectional area of the ligament must be provided for ligaments of type FUNCTION.")
        elif ligament_type in [LigamentType.CONSTANT, LigamentType.LINEAR_SPRING, LigamentType.SECOND_ORDER_SPRING]:
            # You are using a biorbd-like ligament
            if stiffness is None:
                raise ValueError("The stiffness of the ligament must be provided for ligaments of type CONSTANT, LINEAR_SPRING, or SECOND_ORDER_SPRING.")
            if damping is None:
                damping = 0.0

        self.name = name
        self.ligament_type = ligament_type
        self.origin_position = origin_position
        self.insertion_position = insertion_position
        self.stiffness = stiffness
        self.ligament_slack_length = ligament_slack_length
        self.damping = damping
        self.force_length_function = force_length_function
        self.pcsa = pcsa

        # For convenience
        self.origin_parent_name = self.origin_position.parent_name
        self.insertion_parent_name = self.insertion_position.parent_name

    @property
    def name(self) -> str:
        return self._name

    @name.setter
    def name(self, value: str):
        self._name = value

    @property
    def ligament_type(self) -> LigamentType:
        return self._ligament_type

    @ligament_type.setter
    def ligament_type(self, value: LigamentType | str):
        if isinstance(value, str):
            value = LigamentType(value)
        self._ligament_type = value

    @property
    def origin_position(self) -> ViaPointReal:
        return self._origin_position

    @origin_position.setter
    def origin_position(self, value: ViaPointReal):
        if value is None:
            self._origin_position = None
        else:
            self._origin_position = value

    @property
    def insertion_position(self) -> ViaPointReal:
        return self._insertion_position

    @insertion_position.setter
    def insertion_position(self, value: ViaPointReal):
        if value is None:
            self._insertion_position = None
        else:
            self._insertion_position = value

    @property
    def stiffness(self) -> float:
        return self._stiffness

    @stiffness.setter
    def stiffness(self, value: float):
        if value is not None and value <= 0:
            raise ValueError("The maximal force of the force must be greater than 0.")
        if isinstance(value, np.ndarray):
            if value.shape == (1,):
                value = value[0]
            else:
                raise ValueError("The maximal force must be a float.")
        self._stiffness = value

    @property
    def ligament_slack_length(self) -> float:
        return self._ligament_slack_length

    @ligament_slack_length.setter
    def ligament_slack_length(self, value: float):
        if value is not None and value <= 0:
            raise ValueError("The ligament slack length of the force must be greater than 0.")
        if isinstance(value, np.ndarray):
            if value.shape == (1,):
                value = value[0]
            else:
                raise ValueError("The ligament slack length must be a float.")
        self._ligament_slack_length = value

    @property
    def damping(self) -> float:
        return self._damping

    @damping.setter
    def damping(self, value: float):
        if value is not None and value <= 0:
            raise ValueError("The damping of the ligament must be greater than 0.")
        if isinstance(value, np.ndarray):
            if value.shape == (1,):
                value = value[0]
            else:
                raise ValueError("The damping must be a float.")
        self._damping = value

    @property
    def force_length_function(self) -> float:
        return self._force_length_function

    @force_length_function.setter
    def force_length_function(self, value: InterpolationFunction):
        if not isinstance(value, InterpolationFunction):
            raise ValueError("The force_length_function of the ligament must be an InterpolationFunction.")
        self._force_length_function = value

    @property
    def pcsa(self) -> float:
        return self._pcsa

    @pcsa.setter
    def pcsa(self, value: float):
        if value is not None and value <= 0:
            raise ValueError("The pcsa of the ligament must be greater than 0.")
        if isinstance(value, np.ndarray):
            if value.shape == (1,):
                value = value[0]
            else:
                raise ValueError("The pcsa must be a float.")
        self._pcsa = value

    def approximate_ligament(
            self,
            ligament_type: LigamentType,
            plot_approximation: bool = False,
    ) -> "LigamentReal":

        if self.ligament_type == LigamentType.FUNCTION:
            if ligament_type == LigamentType.FUNCTION:
                return self
            else:
                x_points = self.force_length_function.x_points
                y_points = self.force_length_function.y_points
                approximating_function = LIGAMENT_FUNCTION[ligament_type]
                popt, _ = curve_fit(approximating_function, x_points, y_points)

                new_ligament = LigamentReal(
                    name=self.name,
                    ligament_type=ligament_type,
                    origin_position=self.origin_position,
                    insertion_position=self.insertion_position,
                    ligament_slack_length=self.ligament_slack_length,
                    stiffness=popt[0] if ligament_type == LigamentType.LINEAR_SPRING else popt[:2] if ligament_type == LigamentType.SECOND_ORDER_SPRING else None,
                    damping=self.damping,
                )

                if plot_approximation:
                    plt.figure()
                    x_vector = np.linspace(min(x_points), max(x_points), 100)
                    plt.plot(x_points, y_points, "ok", label="Original points")
                    plt.plot(x_vector, self.force_length_function.evaluate(x_vector), "-k", label="Original function")
                    plt.plot(x_vector, approximating_function(x_vector, *popt), "--r", label=f"Approximating function of type {ligament_type.value}")
                    plt.xlabel("Ligament length")
                    plt.ylabel("Ligament force")
                    plt.title(f"Approximation of the ligament {self.name} with a function of type {ligament_type.value}")
                    plt.legend()
                    plt.show()

        else:
            if ligament_type == LigamentType.FUNCTION:
                x_vector = np.linspace(0, 1, 100)
                if self.ligament_type == LigamentType.CONSTANT:
                    y_vector =
                elif self.ligament_type == LigamentType.LINEAR_SPRING:
                    approximating_function = lambda x, stiffness: - stiffness * (x - self.ligament_slack_length)
                elif self.ligament_type == LigamentType.SECOND_ORDER_SPRING:
                    approximating_function = lambda x, stiffness: - stiffness[0] * (x - self.ligament_slack_length) - \
                                                                  stiffness[1] * (x - self.ligament_slack_length) ** 2

            popt, _ = curve_fit(approximating_function, x_points, y_points)

            new_ligament = LigamentReal(
                name=self.name,
                ligament_type=ligament_type,
                origin_position=self.origin_position,
                insertion_position=self.insertion_position,
                ligament_slack_length=self.ligament_slack_length,
                stiffness=popt[0] if ligament_type == LigamentType.LINEAR_SPRING else popt[
                    :2] if ligament_type == LigamentType.SECOND_ORDER_SPRING else None,
                damping=self.damping,
            )

            if plot_approximation:
                plt.figure()
                x_vector = np.linspace(min(x_points), max(x_points), 100)
                plt.plot(x_points, y_points, "ok", label="Original points")
                plt.plot(x_vector, self.force_length_function.evaluate(x_vector), "-k", label="Original function")
                plt.plot(x_vector, approximating_function(x_vector, *popt), "--r",
                         label=f"Approximating function of type {ligament_type.value}")
                plt.xlabel("Ligament length")
                plt.ylabel("Ligament force")
                plt.title(f"Approximation of the ligament {self.name} with a function of type {ligament_type.value}")
                plt.legend()
                plt.show()

        return new_ligament

    def to_biomod(self) -> str:
        """
        Define the print function, so it automatically formats things in the .bioMod file properly
        """

        if self.ligament_type == LigamentType.FUNCTION:
            raise NotImplementedError("The to_biomod method is not implemented for ligaments of type FUNCTION."
                                      "Please use model.approximate_ligaments(ligament_type, damping), with ligament_type in LigamentType.CONSTANT, LigamentType.LINEAR_SPRING, or LigamentType.SECOND_ORDER_SPRING.")

        out_string = f"ligament\t{self.name}\n"
        out_string += f"\ttype\t{self.ligament_type.value}\n"
        out_string += f"\toriginposition\t{np.round(self.origin_position.position[0, 0], 4)}\t{np.round(self.origin_position.position[1, 0], 4)}\t{np.round(self.origin_position.position[2, 0], 4)}\n"
        out_string += f"\tinsertionposition\t{np.round(self.insertion_position.position[0, 0], 4)}\t{np.round(self.insertion_position.position[1, 0], 4)}\t{np.round(self.insertion_position.position[2, 0], 4)}\n"
        out_string += f"\tstiffness\t{self.stiffness:0.4f}\n"
        out_string += f"\tligamentslacklength\t{self.ligament_slack_length:0.4f}\n"
        out_string += f"\tdamping\t{self.damping:0.4f}\n"
        out_string += "endligament\n"
        out_string += "\n\n"

        return out_string

    def to_osim(self) -> etree.Element:
        """
        Generate OpenSim XML representation of the ligament
        """

        if self.ligament_type in [LigamentType.CONSTANT, LigamentType.LINEAR_SPRING, LigamentType.SECOND_ORDER_SPRING]:
            raise NotImplementedError("The to_osim method is not implemented for ligaments of types LigamentType.CONSTANT, LigamentType.LINEAR_SPRING, LigamentType.SECOND_ORDER_SPRING."
                                      "Please use model.approximate_ligaments(ligament_type=LigamentType.FUNCTION).")

        ligament_elem = etree.Element("Ligament", name=self.name)

        resting_length = etree.SubElement(ligament_elem, "resting_length")
        resting_length.text = f"{self.ligament_slack_length:.8f}"

        pcsa_force = etree.SubElement(ligament_elem, "pcsa_force")
        pcsa_force.text = f"{self.ligament_slack_length:.8f}"

        # Geometry path
        geometry_path = etree.SubElement(ligament_elem, "GeometryPath", name="path")
        path_point_set = etree.SubElement(geometry_path, "PathPointSet")
        path_objects = etree.SubElement(path_point_set, "objects")

        # Origin
        origin_elem = self.origin_position.to_osim()
        if origin_elem is not None:
            origin_elem.set("name", f"{self.name}_origin")
            path_objects.append(origin_elem)
        else:
            raise ValueError(f"The origin position of the ligament {self.name} has to be defined.")

        # Insertion
        insertion_elem = self.insertion_position.to_osim()
        if insertion_elem is not None:
            insertion_elem.set("name", f"{self.name}_insertion")
            path_objects.append(insertion_elem)
        else:
            raise ValueError(f"The insertion position of the ligament {self.name} has to be defined.")

        # Force length curve
        force_length_curve = self.force_length_function.to_osim()
        if force_length_curve is not None:
            force_length_curve.set("name", f"force_length_curve")
            path_objects.append(force_length_curve)
        else:
            raise ValueError(f"The force length curve of the ligament {self.name} has to be defined.")

        return ligament_elem
