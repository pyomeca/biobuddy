import biorbd
import numpy as np
from plotly.subplots import make_subplots
from math import ceil
import plotly.graph_objects as go


def main():
    """
    Plot 3 graphs to check the capability of muscles within a musculoskeletal model (.bioMod model). Plot the max strength of each
    muscle, their moment arm over each joint, and the max torque that each muscle can apply on each joint.

    Only the model path is required to perform the check. The muscles parameters will by default automatically be computed
    over the entire range of motion of each joint. Optionally, a custom range for the joints can be provided in the
    states_from_model_ranges function.

    If you want to add additional passive joint forces in the joint torque computation that aren't directly defined in
    the .bioMod file but are computed later on in your simulations, it is possible to do so in the indicated line in
    the compute_torques function.
    """

    model_path = "wholebody_reference.bioMod"  # Path to the model to check

    states = states_from_model_ranges(model_path)
    muscle_max_force, muscle_min_force = compute_muscle_forces(model_path, states)
    muscle_lengths = compute_muscle_lengths(model_path, states)
    muscle_optimal_lengths = return_optimal_lengths(model_path)
    muscle_moment_arm = compute_moment_arm(model_path, states)
    muscle_max_torque, muscle_min_torque = compute_torques(model_path, states)
    plot_force_length(
        model_path,
        states,
        muscle_max_force,
        muscle_min_force,
        muscle_lengths,
        muscle_optimal_lengths,
    )
    plot_moment_arm(model_path, states, muscle_moment_arm)
    plot_torques(model_path, states, muscle_max_torque, muscle_min_torque)


def states_from_model_ranges(
    model_path: str, nb_states: int = 50, custom_ranges: np.ndarray = None
) -> np.ndarray:
    """
    Create an array of model states (position vector q) from the model max and min ranges or from custom ranges

    Parameters
    ----------
    model_path: str
        Path to the model
    nb_states: int = 50
        Number of states between the min and max ranges
    custom_ranges: np.ndarray = None
        Custom ranges between min and max ranges used for the states

    Returns
    -------
    states: np.ndarray
        Model states
    """
    if custom_ranges is None:
        model = biorbd.Model(model_path)
        ranges = biorbd.get_range_q(model)
    else:
        ranges = custom_ranges
    states = []
    for joint_idx in range(len(ranges[0])):
        joint_array = np.linspace(ranges[0][joint_idx], ranges[1][joint_idx], nb_states)
        states.append(joint_array)
    return np.array(states)


def compute_muscle_forces(
    model_path: str, states: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute for each muscle the max force (muscle activation = 1) and the min force (muscle activation = 0) over every states

    Parameters
    ----------
    model_path: str
        Path to the model
    states: np.ndarray
        Model states

    Returns
    -------
    model_max_force: np.ndarray
        Max muscle forces for every state
    model_min_force: np.ndarray
        Min muscle forces for every state
    """
    model = biorbd.Model(model_path)
    nb_muscles = len(model.muscleNames())
    nb_dof = model.nbQ()
    nb_frame = states.shape[1]
    muscle_states = model.stateSet()
    model_max_force = np.ndarray((nb_muscles, nb_frame))
    model_min_force = np.ndarray((nb_muscles, nb_frame))
    for i in range(nb_frame):
        q = states[:, i]
        qdot = np.zeros(nb_dof)  # Default the speed at 0
        model.updateMuscles(q)
        # Compute max force array
        for state in muscle_states:
            state.setActivation(1)
        model_max_force_array = (
            model.muscleForces(muscle_states, q, qdot).to_array().copy()
        )
        # Compute min force array
        for state in muscle_states:
            state.setActivation(0)
        model_min_force_array = (
            model.muscleForces(muscle_states, q, qdot).to_array().copy()
        )
        for m in range(nb_muscles):
            model_max_force[m, i] = model_max_force_array[m]
            model_min_force[m, i] = model_min_force_array[m]
    return model_max_force, model_min_force


def compute_muscle_lengths(model_path: str, states: np.ndarray) -> np.ndarray:
    """
    Compute muscle lengths for every state

    Parameters
    ----------
    model_path: str
        Path to the model
    states: np.ndarray
        Model states

    Returns
    -------
    muscle_lengths: np.ndarray
        Muscle lengths for every state
    """
    model = biorbd.Model(model_path)
    nb_muscles = len(model.muscleNames())
    nb_frame = states.shape[1]
    muscle_length = np.zeros((nb_muscles, nb_frame))
    for i in range(nb_frame):
        q = states[:, i]
        for m in range(nb_muscles):
            model.updateMuscles(q, True)
            muscle_length[m, i] = model.muscle(m).length(model, q, True)
    return muscle_length


def return_optimal_lengths(model_path: str) -> np.ndarray:
    """
    Fetch muscles optimal lengths for every state

    Parameters
    ----------
    model_path: str
        Path to the model

    Returns
    -------
    muscle_optimal_lengths: np.ndarray
        Muscle optimal lengths for every state
    """
    model = biorbd.Model(model_path)
    nb_muscles = len(model.muscleNames())
    muscle_optimal_lengths = np.zeros(nb_muscles)
    for m in range(nb_muscles):
        muscle_optimal_lengths[m] = model.muscle(m).characteristics().optimalLength()
    return muscle_optimal_lengths


def compute_moment_arm(model_path: str, states: np.ndarray) -> np.ndarray:
    """
    Compute muscle moment arms for every joint in every state

    Parameters
    ----------
    model_path: str
        Path to the model
    states: np.ndarray
        Model states

    Returns
    -------
    muscle_moment_arm: np.ndarray
     Muscle moment arm for every state
    """
    model = biorbd.Model(model_path)
    nb_muscles = len(model.muscleNames())
    nb_frame = states.shape[1]
    nb_dof = model.nbQ()
    muscle_moment_arm = np.ndarray((nb_dof, nb_muscles, nb_frame))
    for i in range(nb_frame):
        bio_moment_arm_array = model.musclesLengthJacobian(states[:, i]).to_array()
        for m in range(nb_muscles):
            muscle_moment_arm[:, m, i] = bio_moment_arm_array[m]
    return muscle_moment_arm


def compute_torques(
    model_path: str, states: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute for every state the minimal torque applied on every joint when no muscles are activated and the maximal
    torque when one muscle is activated for every individual muscle

    Parameters
    ----------
    model_path: str
        Path to the model
    states: np.ndarray
        Model states

    Returns
    -------
    joint_max_torques: np.ndarray
        Max joint torque for every state when one muscle is activated
    joint_min_torques: np.ndarray
        Min joint torque for every state
    """
    model = biorbd.Model(model_path)
    nb_muscles = len(model.muscleNames())
    nb_frame = states.shape[1]
    nb_dof = model.nbQ()
    muscle_states = model.stateSet()
    qdot = np.zeros(nb_dof)
    joint_max_torques = np.ndarray((nb_dof, nb_muscles, nb_frame))
    joint_min_torques = np.ndarray((nb_dof, nb_frame))
    for i in range(nb_frame):
        q = states[:, i]
        model.updateMuscles(q)
        # Compute max torque for each joint with only one activated muscle
        for m in range(nb_muscles):
            for state in range(len(muscle_states)):
                if state == m:
                    muscle_states[state].setActivation(1)
                else:
                    muscle_states[state].setActivation(0)
            # If you wish to add custom passive torques to your model and have them accounted for in the check, add them here
            joint_max_torques[:, m, i] = (
                model.muscularJointTorque(muscle_states, q, qdot).to_array().copy()
            )
    # Compute min torques
    for state in muscle_states:
        state.setActivation(0)
    for i in range(nb_frame):
        q = states[:, i]
        model.updateMuscles(q)
        # If you wish to add custom passive torques to your model and have them accounted for in the check, add them here
        joint_min_torques[:, i] = (
            model.muscularJointTorque(muscle_states, q, qdot).to_array().copy()
        )
    return joint_max_torques, joint_min_torques


def plot_force_length(
    model_path: str,
    states: np.ndarray,
    muscle_max_force: np.ndarray,
    muscle_min_force: np.ndarray,
    muscle_lengths: np.ndarray,
    muscle_optimal_lengths: np.ndarray,
) -> None:
    """
    Plot force lengths graphs for the model using plotly

    Parameters
    ----------
    model_path: str
        Path to the model
    states: np.ndarray
        Model states
    muscle_max_force: np.ndarray
        Muscle max force array
    muscle_min_force: np.ndarray
        Muscle min force array
    muscle_lengths: np.ndarray
        Muscle optimal lengths array
    muscle_optimal_lengths: np.ndarray
        Muscle optimal lengths array
    """
    model = biorbd.Model(model_path)
    nb_muscles = len(model.muscleNames())
    nb_frame = states.shape[1]
    nb_lines = 1
    muscle_names = []
    for muscle in range(nb_muscles):
        muscle_names.append(f"{model.muscleNames()[muscle].to_string()}")
    fig = make_subplots(
        rows=nb_lines, cols=2, subplot_titles=["muscle_Forces", "muscle_Lengths"]
    )
    row = 1
    visible_arg = [False] * nb_muscles * 4

    for muscle in range(nb_muscles):
        col = 1
        x = np.linspace(1, nb_frame, nb_frame)
        fig.add_trace(
            go.Scatter(
                x=x,
                y=muscle_max_force[muscle, :],
                name=muscle_names[muscle] + "_Max_Force",
            ),
            row=row,
            col=col,
        )
        fig.add_trace(
            go.Scatter(
                x=x,
                y=muscle_min_force[muscle, :],
                name=muscle_names[muscle] + "_Min_Force",
            ),
            row=row,
            col=col,
        )
        col += 1
        fig.add_trace(
            go.Scatter(
                x=x,
                y=muscle_lengths[muscle, :],
                name=muscle_names[muscle] + "_Length",
            ),
            row=row,
            col=col,
        )
        fig.add_trace(
            go.Scatter(
                x=x,
                y=np.full(nb_frame, muscle_optimal_lengths[muscle]),
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
        ]
    )

    fig.show()


def plot_moment_arm(
    model_path: str, states: np.ndarray, muscle_moment_arm: np.ndarray
) -> None:
    """
    Plot moment arm for each muscle of the model over each joint using plotly

    Parameters
    ----------
    model_path: str
        Path to the model
    states: np.ndarray
        Model states
    muscle_moment_arm: np.ndarray
        Muscle moment arm array
    """
    model = biorbd.Model(model_path)
    nb_muscles = len(model.muscleNames())
    nb_dof = model.nbQ()
    var = ceil(nb_muscles / 5)
    nb_line = var if var < 5 else 5
    muscle_names = []
    for muscle in range(nb_muscles):
        muscle_names.append(f"{model.muscleNames()[muscle].to_string()}")
    dof_names = []
    for dof in range(nb_dof):
        dof_names.append(model.nameDof()[dof].to_string())

    fig = make_subplots(
        rows=nb_line,
        cols=ceil(nb_muscles / nb_line),
        subplot_titles=tuple(muscle_names),
    )

    visible_arg = [False] * nb_dof * nb_muscles

    for dof in range(nb_dof):
        row = 0
        x = states[dof, :]
        for muscle in range(nb_muscles):
            col = muscle % ceil(nb_muscles / nb_line) + 1
            if col == 1:
                row = row + 1
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=muscle_moment_arm[dof, muscle, :],
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
        updatemenus=[
            go.layout.Updatemenu(
                active=0,
                buttons=list(
                    map(lambda dof_name: create_layout_button_kin(dof_name), dof_names)
                ),
            )
        ]
    )

    fig.show()


def plot_torques(
    model_path: str,
    states: np.ndarray,
    muscle_max_torques: np.ndarray,
    muscle_min_torques: np.ndarray,
) -> None:
    """
    Plot the min and max torques at each joint of the model for each muscle activation using plotly

    Parameters
    ----------
    model_path: str
        Path to the model
    states: np.ndarray
        Model states
    muscle_max_torques: np.ndarray
        Muscle max torques array
    muscle_min_torques: np.ndarray
        Muscle min torques array
    """
    model = biorbd.Model(model_path)
    nb_muscles = len(model.muscleNames())
    nb_dof = model.nbQ()
    var = ceil(nb_muscles / 5)
    nb_line = var if var < 5 else 5
    muscle_names = []
    for muscle in range(nb_muscles):
        muscle_names.append(f"{model.muscleNames()[muscle].to_string()}")
    dof_names = []
    for dof in range(nb_dof):
        dof_names.append(model.nameDof()[dof].to_string())

    fig = make_subplots(
        rows=nb_line,
        cols=ceil(nb_muscles / nb_line),
        subplot_titles=tuple(muscle_names),
    )

    visible_arg = [False] * nb_dof * nb_muscles * 2

    for dof in range(nb_dof):
        row = 0
        x = states[dof, :]
        for muscle in range(nb_muscles):
            col = muscle % ceil(nb_muscles / nb_line) + 1
            if col == 1:
                row = row + 1
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=muscle_max_torques[dof, muscle, :],
                    name=muscle_names[muscle] + "_Max_Torque",
                ),
                row=row,
                col=col,
            )
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=muscle_min_torques[dof, :],
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
        updatemenus=[
            go.layout.Updatemenu(
                active=0,
                buttons=list(
                    map(lambda dof_name: create_layout_button_kin(dof_name), dof_names)
                ),
            )
        ]
    )

    fig.show()


if __name__ == "__main__":
    main()
