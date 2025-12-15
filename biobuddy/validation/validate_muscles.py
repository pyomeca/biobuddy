import numpy as np
from math import ceil
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import plotly.graph_objects as go

from ..components.real.biomechanical_model_real import BiomechanicalModelReal


class MuscleValidator:
    def __init__(
        self,
        model: BiomechanicalModelReal,
        nb_states: int = 50,
        custom_ranges: np.ndarray = None,
    ):
        """
        For now only biomodable models are supported, but this class should be extended to support other model types.

        Parameters
        ----------
        model: str
            The model to check the muscles for
        nb_states: int = 50
            Number of states between the min and max ranges
        custom_ranges: np.ndarray = None
            Custom ranges between min and max ranges used for the states
        """
        # Original attributes
        self.model = model
        self.nb_states = nb_states
        self.custom_ranges = custom_ranges

        # Check that the model is correct
        try:
            self.model.to_biomod("temporary.bioMod", with_mesh=False)
        except Exception as e:
            raise NotImplementedError(
                f"Only biorbd is supported as a backend for now. If you need other dynamics "
                f"engines, please contact the developers. "
                f"The model provided is not biomodable: {e}."
            )
        if model.nb_q == 0:
            raise ValueError(
                "Your model has no degrees of freedom. Please provide a model with at least one degree of freedom."
            )
        if model.nb_muscles == 0:
            raise ValueError("Your model has no muscles. Please provide a model with at least one muscle.")

        # Initialize the quantities that will be needed for the plots
        self.states: np.ndarray = self.states_from_model_ranges()
        muscle_max_force, muscle_min_force = self.compute_muscle_forces()
        self.muscle_max_force: np.ndarray = muscle_max_force
        self.muscle_min_force: np.ndarray = muscle_min_force
        self.muscle_lengths: np.ndarray = self.compute_muscle_lengths()
        self.muscle_optimal_lengths: np.ndarray = self.return_optimal_lengths()
        self.muscle_moment_arm: np.ndarray = self.compute_moment_arm()
        muscle_max_torque, muscle_min_torque = self.compute_torques()
        self.muscle_max_torque: np.ndarray = muscle_max_torque
        self.muscle_min_torque: np.ndarray = muscle_min_torque

    def states_from_model_ranges(self) -> np.ndarray:
        """
        Create an array of model states (position vector q) from the model max and min ranges or from custom ranges

        Returns
        -------
        states: np.ndarray
            Model states
        """
        if self.custom_ranges is None:
            ranges = self.model.get_dof_ranges()
        else:
            ranges = self.custom_ranges

        # Check the shape of ranges
        if ranges.size == 0:
            ranges = np.array([[-np.pi] * self.model.nb_q, [np.pi] * self.model.nb_q])
        elif ranges.shape != (2, self.model.nb_q):
            raise NotImplementedError(
                f"Either all ranges or no ranges could be provided for now. Expected shape 2 x {self.model.nb_q}."
                f"If you fall on this error please contact the developers."
            )

        # Set the states as a linear interpolation between the min and max ranges
        states = []
        for joint_idx in range(len(ranges[0])):
            joint_array = np.linspace(ranges[0][joint_idx], ranges[1][joint_idx], self.nb_states)
            states.append(joint_array)

        return np.array(states)

    def compute_muscle_forces(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute for each muscle the max force (muscle activation = 1) and the min force (muscle activation = 0) over every states

        Returns
        -------
        model_max_force: np.ndarray
            Max muscle forces for every state
        model_min_force: np.ndarray
            Min muscle forces for every state
        """
        import biorbd

        nb_muscles = self.model.nb_muscles
        nb_dof = self.model.nb_q

        # TODO: change this to allow for other dynamics engines
        biorbd_model = biorbd.Model("temporary.bioMod")

        muscle_states = biorbd_model.stateSet()
        model_max_force = np.ndarray((nb_muscles, self.nb_states))
        model_min_force = np.ndarray((nb_muscles, self.nb_states))
        for i_frame in range(self.nb_states):
            q = self.states[:, i_frame]
            qdot = np.zeros((nb_dof,))  # Default the speed at 0
            biorbd_model.updateMuscles(q)

            # Compute max force array
            for state in muscle_states:
                state.setActivation(1)
            model_max_force[:, i_frame] = biorbd_model.muscleForces(muscle_states, q, qdot).to_array().copy()

            # Compute min force array
            for state in muscle_states:
                state.setActivation(0)
            model_min_force[:, i_frame] = biorbd_model.muscleForces(muscle_states, q, qdot).to_array().copy()

        return model_max_force, model_min_force

    def compute_muscle_lengths(self) -> np.ndarray:
        """
        Compute muscle lengths for every state

        Returns
        -------
        muscle_lengths: np.ndarray
            Muscle lengths for every state
        """
        import biorbd

        # TODO: change this to allow for other dynamics engines
        biorbd_model = biorbd.Model("temporary.bioMod")

        muscle_length = np.zeros((self.model.nb_muscles, self.nb_states))
        for i_frame in range(self.nb_states):
            q = self.states[:, i_frame]
            for i_muscle in range(self.model.nb_muscles):
                biorbd_model.updateMuscles(q, True)
                muscle_length[i_muscle, i_frame] = biorbd_model.muscle(i_muscle).length(biorbd_model, q, True)
        return muscle_length

    def return_optimal_lengths(self) -> np.ndarray:
        """
        Fetch muscles optimal lengths for every state

        Returns
        -------
        muscle_optimal_lengths: np.ndarray
            Muscle optimal lengths for every state
        """
        muscle_optimal_lengths = np.zeros((self.model.nb_muscles,))
        i_muscle = 0
        for muscle_group in self.model.muscle_groups:
            for muscle in muscle_group.muscles:
                muscle_optimal_lengths[i_muscle] = muscle.optimal_length
                i_muscle += 1
        return muscle_optimal_lengths

    def compute_moment_arm(self) -> np.ndarray:
        """
        Compute muscle moment arms for every joint in every state

        Returns
        -------
        muscle_moment_arm: np.ndarray
         Muscle moment arm for every state
        """
        import biorbd

        # TODO: change this to allow for other dynamics engines
        biorbd_model = biorbd.Model("temporary.bioMod")

        muscle_moment_arm = np.ndarray((self.model.nb_q, self.model.nb_muscles, self.nb_states))
        for i in range(self.nb_states):
            bio_moment_arm_array = biorbd_model.musclesLengthJacobian(self.states[:, i]).to_array()
            for m in range(self.model.nb_muscles):
                muscle_moment_arm[:, m, i] = bio_moment_arm_array[m]
        return muscle_moment_arm

    def compute_torques(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute for every state the minimal torque applied on every joint when no muscles are activated and the maximal
        torque when one muscle is activated for every individual muscle

        Returns
        -------
        joint_max_torques: np.ndarray
            Max joint torque for every state when one muscle is activated
        joint_min_torques: np.ndarray
            Min joint torque for every state
        """
        import biorbd

        # TODO: change this to allow for other dynamics engines
        biorbd_model = biorbd.Model("temporary.bioMod")

        muscle_states = biorbd_model.stateSet()
        qdot = np.zeros((self.model.nb_q,))  # TODO: add quaternion handling
        joint_max_torques = np.ndarray((self.model.nb_q, self.model.nb_muscles, self.nb_states))
        joint_min_torques = np.ndarray((self.model.nb_q, self.nb_states))
        for i_frame in range(self.nb_states):
            q = self.states[:, i_frame]
            biorbd_model.updateMuscles(q)

            # Compute max torque for each joint with only one activated muscle
            for i_muscle in range(self.model.nb_muscles):
                for state in range(len(muscle_states)):
                    if state == i_muscle:
                        muscle_states[state].setActivation(1)
                    else:
                        muscle_states[state].setActivation(0)
                # If you wish to add custom passive torques to your model and have them accounted for in the check, add them here
                joint_max_torques[:, i_muscle, i_frame] = (
                    biorbd_model.muscularJointTorque(muscle_states, q, qdot).to_array().copy()
                )

            # Compute min torques for each joint with only one activated muscle
            for state in muscle_states:
                state.setActivation(0)
            # If you wish to add custom passive torques to your model and have them accounted for in the check, add them here
            joint_min_torques[:, i_frame] = biorbd_model.muscularJointTorque(muscle_states, q, qdot).to_array().copy()

        return joint_max_torques, joint_min_torques

    def plot_force_length(
        self,
    ) -> "go.Figure":
        """
        Plot force lengths graphs for the model using plotly
        """
        from plotly.subplots import make_subplots
        import plotly.graph_objects as go

        nb_muscles = self.model.nb_muscles
        nb_lines = 1
        muscle_names = self.model.muscle_names
        fig = make_subplots(rows=nb_lines, cols=2, subplot_titles=["muscle_Forces", "muscle_Lengths"])
        row = 1
        visible_arg = [False] * nb_muscles * 4
        x = np.linspace(min(self.states[:, 0]), max(self.states[:, -1]), self.nb_states)

        for muscle in range(nb_muscles):
            col = 1
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=self.muscle_max_force[muscle, :],
                    name=muscle_names[muscle] + "_Max_Force",
                ),
                row=row,
                col=col,
            )
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=self.muscle_min_force[muscle, :],
                    name=muscle_names[muscle] + "_Min_Force",
                ),
                row=row,
                col=col,
            )
            col += 1
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=self.muscle_lengths[muscle, :],
                    name=muscle_names[muscle] + "_Length",
                ),
                row=row,
                col=col,
            )
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=np.full(self.nb_states, self.muscle_optimal_lengths[muscle]),
                    name=muscle_names[muscle] + "_Optimal_Length",
                ),
                row=row,
                col=col,
            )

        def create_layout_button_kin(muscle_name):
            muscle_idx = muscle_names.index(muscle_name)
            visible = visible_arg.copy()
            for idx in range(4):
                visible[muscle_idx * 4 + idx] = True
            button = dict(
                label=muscle_name,
                method="update",
                args=[{"visible": visible, "title": muscle_name, "showlegend": True}],
            )
            return button

        fig.update_layout(
            title="Muscular Force–Length",
            title_x=0.5,
            updatemenus=[
                go.layout.Updatemenu(
                    active=0,
                    buttons=list(
                        map(
                            lambda muscle_name: create_layout_button_kin(muscle_name),
                            muscle_names,
                        )
                    ),
                )
            ],
        )

        fig.update_xaxes(title_text="Range (rad)", row=1, col=1)
        fig.update_yaxes(title_text="Force (N)", row=1, col=1)

        fig.update_xaxes(title_text="Range (rad)", row=1, col=2)
        fig.update_yaxes(title_text="Length (m)", row=1, col=2)

        fig.show()
        return fig

    def plot_moment_arm(self) -> "go.Figure":
        """
        Plot moment arm for each muscle of the model over each joint using plotly
        """
        from plotly.subplots import make_subplots
        import plotly.graph_objects as go

        nb_muscles = self.model.nb_muscles
        nb_dof = self.model.nb_q
        muscle_names = self.model.muscle_names
        dof_names = self.model.dof_names

        var = ceil(nb_muscles / 5)
        nb_row = var if var < 5 else 5
        nb_col = ceil(nb_muscles / nb_row)

        fig = make_subplots(
            rows=nb_row,
            cols=nb_col,
            subplot_titles=tuple(muscle_names),
        )

        visible_arg = [False] * nb_dof * nb_muscles

        for dof in range(nb_dof):
            row = 0
            x = self.states[dof, :]
            for muscle in range(nb_muscles):
                col = muscle % nb_col + 1
                if col == 1:
                    row = row + 1
                fig.add_trace(
                    go.Scatter(
                        x=x,
                        y=self.muscle_moment_arm[dof, muscle, :],
                        name=muscle_names[muscle] + "_Moment_Arm",
                    ),
                    row=row,
                    col=col,
                )

        def create_layout_button_kin(dof_name):
            dof_idx = dof_names.index(dof_name)
            visible = visible_arg.copy()
            for idx in range(nb_muscles):
                visible[dof_idx * nb_muscles + idx] = True
            button = dict(
                label=dof_name,
                method="update",
                args=[{"visible": visible, "title": dof_name, "showlegend": True}],
            )
            return button

        fig.update_layout(
            title="Moment arm",
            title_x=0.5,
            updatemenus=[
                go.layout.Updatemenu(
                    active=0,
                    buttons=list(
                        map(
                            lambda dof_name: create_layout_button_kin(dof_name),
                            dof_names,
                        )
                    ),
                )
            ],
        )

        # legends
        for idx_row in range(nb_row):
            fig.update_yaxes(title_text="Moment arm (m)", row=idx_row + 1, col=1)

        for idx_col in range(nb_col):
            fig.update_xaxes(title_text="Range (rad)", row=nb_row, col=idx_col + 1)  # title_font=dict(size=10)

        if nb_col != nb_row:
            for idx_col in range(nb_col - (nb_row * nb_col - nb_muscles), nb_col):
                fig.update_xaxes(title_text="Range (rad)", row=nb_row - 1, col=idx_col + 1)

        fig.show()
        return fig

    def plot_torques(
        self,
    ) -> "go.Figure":
        """
        Plot the min and max torques at each joint of the model for each muscle activation using plotly
        """
        from plotly.subplots import make_subplots
        import plotly.graph_objects as go

        nb_muscles = self.model.nb_muscles
        nb_dof = self.model.nb_q
        muscle_names = self.model.muscle_names
        dof_names = self.model.dof_names

        var = ceil(nb_muscles / 5)
        nb_row = var if var < 5 else 5
        nb_col = ceil(nb_muscles / nb_row)

        fig = make_subplots(
            rows=nb_row,
            cols=nb_col,
            subplot_titles=tuple(muscle_names),
        )

        visible_arg = [False] * nb_dof * nb_muscles * 2

        for dof in range(nb_dof):
            row = 0
            x = self.states[dof, :]
            for muscle in range(nb_muscles):
                col = muscle % nb_col + 1
                if col == 1:
                    row = row + 1
                fig.add_trace(
                    go.Scatter(
                        x=x,
                        y=self.muscle_max_torque[dof, muscle, :],
                        name=muscle_names[muscle] + "_Max_Torque",
                    ),
                    row=row,
                    col=col,
                )
                fig.add_trace(
                    go.Scatter(
                        x=x,
                        y=self.muscle_min_torque[dof, :],
                        name=muscle_names[muscle] + "_Min_Torque",
                    ),
                    row=row,
                    col=col,
                )

        def create_layout_button_kin(dof_name):
            dof_idx = dof_names.index(dof_name)
            visible = visible_arg.copy()
            for idx in range(nb_muscles * 2):
                visible[dof_idx * nb_muscles * 2 + idx] = True
            button = dict(
                label=dof_name,
                method="update",
                args=[{"visible": visible, "title": dof_name, "showlegend": True}],
            )
            return button

        fig.update_layout(
            title="Torque",
            title_x=0.5,
            updatemenus=[
                go.layout.Updatemenu(
                    active=0,
                    buttons=list(
                        map(
                            lambda dof_name: create_layout_button_kin(dof_name),
                            dof_names,
                        )
                    ),
                )
            ],
        )

        # legends
        for idx_row in range(nb_row):
            fig.update_yaxes(title_text="Torque (N.m)", row=idx_row + 1, col=1)

        for idx_col in range(nb_col):
            fig.update_xaxes(title_text="Range (rad)", row=nb_row, col=idx_col + 1)  # title_font=dict(size=10)

        if nb_col != nb_row:
            for idx_col in range(nb_col - (nb_row * nb_col - nb_muscles), nb_col):
                fig.update_xaxes(title_text="Range (rad)", row=nb_row - 1, col=idx_col + 1)

        fig.show()
        return fig
