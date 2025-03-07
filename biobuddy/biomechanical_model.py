from lxml import etree
from xml.etree import ElementTree
import numpy as np
from time import strftime

from model_readers.osim_reader import OsimReader
from .protocols import Data
from .segment_real import SegmentReal
from .muscle_group import MuscleGroup
from .muscle_real import MuscleReal
from .segment_coordinate_system_real import SegmentCoordinateSystemReal
from .biomechanical_model_real import BiomechanicalModelReal


class BiomechanicalModel:
    def __init__(self):
        self.header = None
        self.warnings = None
        self.gravity = np.array([0.0, 0.0, 9.81])  # default value
        self.segments = {}
        self.muscle_groups = {}
        self.muscles = {}
        self.via_points = {}
        self.model = None

    def set_gravity(self, gravity_vector: np.ndarray[float | int, 3]):
        if not isinstance(gravity_vector, np.ndarray) or gravity_vector.shape != (3,):
            raise ValueError("The gravity vector must be a np.ndarray of shape (3,) like np.array([0, 0, 9.81])")
        if not all(isinstance(x, (int, float)) for x in gravity_vector):
            raise ValueError("All components of the gravity vector must be int of float")
        self.gravity = gravity_vector

    def set_header(self, path: str, publications: str = None, credit: str = None, force_units: str = None, length_units: str = None):
        out_string = ""
        out_string += f"\n// File extracted from {path} on the {strftime('%Y-%m-%d %H:%M')}\n"
        if publications:
            out_string += f"\n// Original file publication : {publications}\n"
        if credit:
            out_string += f"\n// Original file credit : {credit}\n"
        if force_units:
            out_string += f"\n// Force units : {force_units}\n"
        if length_units:
            out_string += f"\n// Length units : {length_units}\n"
        self.header = out_string

    def from_osim(self, osim_path: str) -> BiomechanicalModelReal:
        """
        Read an osim file and create both a generic biomechanical model and a personalized model.

        Parameters
        ----------
        osim_path
            The path to the osim file to read from
        """

        osim_model = OsimReader(osim_path=osim_path, output_model=self)
        osim_model.read()

        self.set_gravity(osim_model.gravity)
        self.set_header(path=osim_path,
                        publications=osim_model.publications,
                        credit=osim_model.credit,
                        force_units=osim_model.force_units,
                        length_units=osim_model.length_units)





    def to_real(self, data: Data) -> BiomechanicalModelReal:
        """
        Collapse the model to an actual personalized biomechanical model based on the generic model and the data
        file (usually a static trial)

        Parameters
        ----------
        data
            The data to collapse the model from
        """
        model = BiomechanicalModelReal()

        model.gravity = self.gravity

        for name in self.segments:
            s = self.segments[name]

            scs = SegmentCoordinateSystemReal()
            if s.segment_coordinate_system is not None:
                scs = s.segment_coordinate_system.to_scs(
                    data,
                    model,
                    model.segments[s.parent_name].segment_coordinate_system if s.parent_name else None,
                )

            inertia_parameters = None
            if s.inertia_parameters is not None:
                inertia_parameters = s.inertia_parameters.to_real(data, model, scs)

            mesh = None
            if s.mesh is not None:
                mesh = s.mesh.to_mesh(data, model, scs)

            mesh_file = None
            if s.mesh_file is not None:
                mesh_file = s.mesh_file.to_mesh_file(data)

            model.segments[s.name] = SegmentReal(
                name=s.name,
                parent_name=s.parent_name,
                segment_coordinate_system=scs,
                translations=s.translations,
                rotations=s.rotations,
                q_ranges=s.q_ranges,
                qdot_ranges=s.qdot_ranges,
                inertia_parameters=inertia_parameters,
                mesh=mesh,
                mesh_file=mesh_file,
            )

            for marker in s.markers:
                model.segments[name].add_marker(marker.to_marker(data, model, scs))

            for contact in s.contacts:
                model.segments[name].add_contact(contact.to_contact(data))

        for name in self.muscle_groups:
            mg = self.muscle_groups[name]

            model.muscle_groups[mg.name] = MuscleGroup(
                name=mg.name,
                origin_parent_name=mg.origin_parent_name,
                insertion_parent_name=mg.insertion_parent_name,
            )

        for name in self.muscles:
            m = self.muscles[name]

            if m.muscle_group not in model.muscle_groups:
                raise RuntimeError(
                    f"Please create the muscle group {m.muscle_group} before putting the muscle {m.name} in it."
                )

            model.muscles[m.name] = m.to_muscle(model, data)

        for name in self.via_points:
            vp = self.via_points[name]

            if vp.muscle_name not in model.muscles:
                raise RuntimeError(
                    f"Please create the muscle {vp.muscle_name} before putting the via point {vp.name} in it."
                )

            if vp.muscle_group not in model.muscle_groups:
                raise RuntimeError(
                    f"Please create the muscle group {vp.muscle_group} before putting the via point {vp.name} in it."
                )

            model.via_points[vp.name] = vp.to_via_point(data)

        return model


    def personalize_model(self, data: Data):
        """
        Collapse the model to an actual personalized biomechanical model based on the generic model and the data
        file (usually a static trial)

        Parameters
        ----------
        data
            The data to collapse the model from
        """
        self.model = self.to_real(data)

    def to_biomod(self, save_path: str):
        """
        Write the .bioMod file

        Parameters
        ----------
        save_path
            The path to save the bioMod
        """
        if self.model is None:
            raise RuntimeError("The model was not created yet. You can create the model using BiomechanicalModel.personalize_model,  BiomechanicalModel.from_osim, or BiomechanicalModel.from_biomod.")

        self.model.to_biomod(save_path)

    def to_osim(self, save_path: str, print_warnings: bool = True):
        """
        Write the .osim file

        Parameters
        ----------
        save_path
            The path to save the osim to
        print_warnings
            If the function should print warnings or not in the osim output file if problems are encountered
        """
        raise NotImplementedError("meh")
