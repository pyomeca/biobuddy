import numpy as np
from pathlib import Path
import biorbd
from collections import defaultdict
import os
import warnings

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from math import isqrt, ceil
from scipy.optimize import root_scalar
import copy

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from biobuddy import BiomechanicalModelReal

# TODO : mettre au prore, fr --> en


class MuscleMomentArmAnalyzer:
    """ """

    POSSIBLE_SIGNS = [-1, 0, 1]

    def __init__(self, model_path: str):
        """
        Initialize the MuscleMomentArmAnalyzer.

        Parameters
        ----------
        model_path : str
            Path to the .bioMod file describing the musculoskeletal model.

        Attributes
        ----------
        model : BiomechanicalModelReal
            The loaded biomechanical model.
        biorbd_model : biorbd.Model
            The biorbd model loaded from the given path.
        sign_lever_arm : dict
            Dictionary to store the expected sign of the moment arm for each DOF and muscle.
        ranges_by_joint : dict
            Computed ranges of moment arm signs for each DOF and muscle.
        accurate_ranges_by_joint : dict
            Ranges filtered by the expected sign (if provided).
        accurate_ranges_array : np.ndarray
            Array of accurate ROM for each DOF.
        """
        self.model = BiomechanicalModelReal().from_biomod(model_path)
        self.biorbd_model = biorbd.Model(model_path)
        self.sign_lever_arm = {}
        self._ranges_by_joint = None  # Lazy evaluation
        self.accurate_ranges_by_joint = {}
        self.accurate_ranges_array = np.zeros((self.model.nb_q, 2))

        # model.get_dof_ranges()

    def create_sign_lever_arm_user(self, list_sign: np.ndarray):
        """
        Create and update the sign_lever_arm dictionary from a user-provided NumPy array.

        Parameters
        ----------
        list_sign : np.ndarray
            A 2D array of shape (nb_q, nb_muscles) where each element is -1, 0, or 1,
            representing the expected sign of the moment arm for each DOF and muscle.

        Raises
        ------
        ValueError
            If the shape of list_sign is incorrect or if any sign is not -1, 0, or 1.
        """
        if list_sign.shape != (self.model.nb_q, self.model.nb_muscles):
            raise ValueError(
                f"Invalid shape, need : ({self.model.nb_q},{self.model.nb_muscles}) but have {list_sign.shape}"
            )
        sign_lever_arm_user = defaultdict(dict)
        for idx_q, q_name in enumerate(self.model.dof_names):
            for idx_m, m_name in enumerate(self.model.muscle_names):
                sign = list_sign[idx_q, idx_m]
                if sign not in self.POSSIBLE_SIGNS:
                    raise ValueError(f"Invalid sign {sign} at ({q_name}, {m_name}). Sign must be either -1, 0 or 1")
                sign_lever_arm_user[q_name][m_name] = sign

        self.update_sign_lever_arm(sign_lever_arm_user)

    def update_sign_lever_arm(self, sign_lever_arm_user: dict):
        """
        Update the sign_lever_arm dictionary with a user-provided dictionary.

        Parameters
        ----------
        sign_lever_arm_user : dict
            A dictionary of the form {dof_name: {muscle_name: sign}} where sign is -1, 0, or 1,
            representing the expected sign of the moment arm for each DOF and muscle.
        """
        check = self.check_sign_lever_arm_gived_by_user(sign_lever_arm_user)
        if check:
            self.sign_lever_arm = sign_lever_arm_user
        else:
            print("Incorrect input. sign_lever_arm is assigned as defaultdict(dict)")
            self.sign_lever_arm = defaultdict(dict)

    def check_sign_lever_arm_gived_by_user(self, sign_lever_arm):
        """
        Check the validity of the user-provided sign_lever_arm dictionary.

        Parameters
        ----------
        sign_lever_arm_user : dict
            A dictionary of the form {dof_name: {muscle_name: sign}} where sign is -1, 0, or 1,
            representing the expected sign of the moment arm for each DOF and muscle.
        """
        check = True
        for dof_name in sign_lever_arm:
            if dof_name not in self.model.dof_names:
                warnings.warn(f"{dof_name} isn't a valid dof name")
                break
            for key in list(sign_lever_arm[dof_name].keys()):
                if key not in self.model.muscle_names:
                    warnings.warn(f"{key}  isn't a valid muscle name")
                    break
                if sign_lever_arm[dof_name][key] not in self.POSSIBLE_SIGNS:
                    warnings.warn(f"Sign must be either -1, 0 or 1, but got {sign_lever_arm[dof_name][key]}")
                    break

        return check

    def compute_joint_states(self, nb_states: int):
        """
        Compute states for every joint, uniformly distributed in the range of motion

        Parameters
        ----------
        nb_states : int
             number of states to compute for each joint
        Returns
        -------
        states : np.ndarray
        """
        joint_states = np.zeros((self.model.nb_q, nb_states))
        for idx_q in range(self.model.nb_q):
            try:
                joint_states[idx_q, :] = np.linspace(
                    self.model.get_dof_ranges()[0, idx_q],
                    self.model.get_dof_ranges()[1, idx_q],
                    nb_states,
                )
            except:
                raise ValueError("Can't get dof ranges. Maybe you forgot to specify ranges in your model file ?")
        return joint_states

    def compute_moment_arm(self, states) -> np.ndarray:
        """
        This method is inspired by the original method `compute_moment_arm`
        in biobuddy/validation/validate_muscles.py file --> MuscleValidator

        Compute muscle moment arms for every joint in every state

        Returns
        -------
        muscle_moment_arm: np.ndarray
        Muscle moment arm for every state
        """

        # TODO: change this to allow for other dynamics engines
        nb_states = states.shape[1]
        muscle_moment_arm = np.ndarray((self.model.nb_q, self.model.nb_muscles, nb_states))
        for i in range(nb_states):
            bio_moment_arm_array = self.biorbd_model.musclesLengthJacobian(states[:, i]).to_array()
            for m in range(self.model.nb_muscles):
                muscle_moment_arm[:, m, i] = bio_moment_arm_array[m]
        return muscle_moment_arm

    def find_zero_newton(self, q_init: np.ndarray, joint_id: int, muscle_id: int):
        """
        Found q* such that r_{joint_id, muscle_id}(q*) = 0
        by only modifying joint_id (Newton 1D).

        Parameters
        ----------
        q_init : (n_q,)
        joint_id : joint index
        muscle_id : muscle index
        """
        q = q_init.copy()

        def f(x):
            q_local = q.copy()
            q_local[joint_id] = x
            J = self.biorbd_model.musclesLengthJacobian(q_local).to_array()
            return J[muscle_id, joint_id]

        sol = root_scalar(f, method="newton", x0=q[joint_id])

        if sol.converged:
            q[joint_id] = sol.root
            return q
        return None

    @property
    def ranges_by_joint(self):
        """
        Lazily compute and cache the ranges_by_joint property.
        """
        if self._ranges_by_joint is None:
            self._ranges_by_joint = self.compute_moment_arm_ranges()
        return self._ranges_by_joint

    def compute_moment_arm_ranges(self, tol: float = 1e-6, nb_states: int = 50):
        """
        Directly returns the intervals of constant sign for the moment arm.

        Returns
        -------
        dict[j][m] -> list of {"range": (a,b), "sign": ±1}
        """

        # Compute moment arms for every state
        states = self.compute_joint_states(nb_states)
        R = self.compute_moment_arm(states)

        R[np.abs(R) < tol] = 0.0

        result = defaultdict(dict)

        n_q = self.model.nb_q
        n_m = self.model.nb_muscles

        for idx_q in range(n_q):
            for idx_m in range(n_m):
                ranges = []

                if np.all(np.abs(R[idx_q, idx_m]) < tol):
                    a = self.model.get_dof_ranges()[0, idx_q]
                    b = self.model.get_dof_ranges()[1, idx_q]
                    ranges.append({"range": (a, b), "sign": 0})

                else:

                    # 1. Detect indices where the sign of the moment arm changes between consecutive states
                    prod = R[idx_q, idx_m, :-1] * R[idx_q, idx_m, 1:]
                    flip_idx = np.where(prod < 0)[0]

                    # 2. Newton --> find zeros more precisely
                    zeros = []
                    for i in flip_idx:
                        q_star = self.find_zero_newton(states[:, i], idx_q, idx_m)
                        if q_star is not None:
                            zeros.append(q_star[idx_q])

                    zeros = sorted(zeros)

                    # 3. bounds
                    bounds = [self.model.get_dof_ranges()[0, idx_q]] + zeros + [self.model.get_dof_ranges()[1, idx_q]]

                    # 4. Determine the sign of the moment arm in each interval between detected zero crossings
                    for a, b in zip(bounds[:-1], bounds[1:]):

                        mid = 0.5 * (a + b)

                        q_test = np.zeros(n_q)
                        q_test[idx_q] = mid

                        r = self.biorbd_model.musclesLengthJacobian(q_test).to_array()[idx_m, idx_q]
                        s = int(np.sign(r))

                        # Explicitly handle zero sign intervals for robustness
                        ranges.append({"range": (a, b), "sign": s})

                result[self.model.dof_names[idx_q]][self.model.muscle_names[idx_m]] = ranges

        return dict(result)

    def merge_ranges_joint(self, idx_q):
        """
        ranges_joint : dict[muscle_idx] -> list of {"range": (a,b), "sign": ±1}

        Returns
        -------
        merged_bounds : list of float
            bornes uniques triées de tous les intervalles de tous les muscles
        """
        bounds = set()

        ranges = self.get_ranges_from_idx_q(idx_q)
        for muscle_ranges in ranges.values():
            for r in muscle_ranges:
                a, b = r["range"]
                bounds.add(a)
                bounds.add(b)

        merged_bounds = sorted(bounds)
        return merged_bounds

    # def ranges_q_muscle(self, idx_q, idx_muscle):
    #     return self.ranges_by_joint[idx_q][idx_muscle]

    def get_ranges_from_idx_q(self, sign_dict: dict, idx_q: int):
        dof_name = self.model.dof_names[idx_q]
        return sign_dict[dof_name]

    def get_ranges_from_idx_q_and_m(self, sign_dict: dict, idx_q: int, idx_m: int):
        dof_name = self.model.dof_names[idx_q]
        muscle_name = self.model.muscle_names[idx_m]
        return sign_dict[dof_name][muscle_name]

    def accurate_ranges_from_true_sign(self):
        if self.sign_lever_arm == {}:
            raise ValueError(
                "No sign_lever_arm. Please, use either create_sign_lever_arm_user() or update_sign_lever_arm to create it"
            )
        accurate_ranges = copy.deepcopy(self.ranges_by_joint)

        for q_name in self.model.dof_names:
            for m_name in self.model.muscle_names:

                all_items = accurate_ranges[q_name][m_name]
                expected_sign = self.sign_lever_arm[q_name][m_name]

                if expected_sign not in [item["sign"] for item in all_items]:
                    warnings.warn(f"There is no range with the sign {expected_sign} " f"for {q_name} {m_name}")
                else:
                    accurate_ranges[q_name][m_name] = [item for item in all_items if item["sign"] == expected_sign]

        print("\nComparison with user sign : ")
        self.compare_ranges_and_user_sign(accurate_ranges)

        self.accurate_ranges_by_joint = accurate_ranges

    def compare_ranges_and_user_sign(self, accurate_ranges: dict):
        """
        Compare the accurate ranges with the user-provided signs and raise errors if there are mismatches.
        Parameters
        ----------
        accurate_ranges : dict[j][m] -> list of {"range": (a,b), "sign": ±1}
             Ranges computed by accurate_ranges_from_true_sign method
        """
        difference = False
        for q_name, muscles_sign in self.sign_lever_arm.items():

            # Check if q_name is in accurate_ranges
            if q_name not in accurate_ranges:
                raise ValueError(f"{q_name} not found in accurate_ranges")

            for m_name, expected_sign in muscles_sign.items():

                # Muscle is missing in accurate_ranges[q_name]
                if m_name not in accurate_ranges[q_name]:

                    if expected_sign == 0:
                        # ok
                        continue
                    else:
                        warnings.warn(
                            f"{m_name} missing in accurate_ranges[{q_name}] " f"but expected sign is {expected_sign}"
                        )
                        difference = True
                        continue

                # Muscle is present in accurate_ranges[q_name], check if expected_sign is among the available signs
                ranges = accurate_ranges[q_name][m_name]
                available_signs = {item["sign"] for item in ranges}

                if expected_sign not in available_signs:
                    warnings.warn(
                        f"Sign mismatch for {q_name}-{m_name}: "
                        f"expected user sign {expected_sign}, "
                        f"available {available_signs}"
                    )
                    difference = True
        if not difference:
            print("Correct")
        return difference

    def create_accurate_rom(self):
        if self.accurate_ranges_by_joint == {}:
            if self.sign_lever_arm == {}:
                print(
                    "No sign_lever_arm. Please, use either create_sign_lever_arm_user() or update_sign_lever_arm to create it"
                )
                return None
            else:
                self.accurate_ranges_from_true_sign()

        for idx_q, q_name in enumerate(self.model.dof_names):
            range = np.array(
                [
                    self.model.get_dof_ranges()[0, idx_q],
                    self.model.get_dof_ranges()[1, idx_q],
                ]
            )
            for m_name in self.model.muscle_names:
                range[0] = max(
                    range[0],
                    self.accurate_ranges_by_joint[q_name][m_name][0]["range"][0],
                )
                range[1] = min(
                    range[1],
                    self.accurate_ranges_by_joint[q_name][m_name][0]["range"][1],
                )

            self.accurate_ranges_array[idx_q, :] = range
        return self.accurate_ranges_array

    def get_correct_part_mvt(self, q):

        if q.shape[0] != self.model.nb_q:
            raise ValueError(f"Incorrect shape, must have {self.model.nb_q} but got {q.shape[0]}")

        if np.allclose(self.accurate_ranges_array, np.zeros((self.model.nb_q, 2))):
            self.create_accurate_rom()

        N = q.shape[1]

        idx_correct = []
        idx_incorrect = []
        # sort idx
        for n in range(N):
            is_correct = np.all(
                (self.accurate_ranges_array[:, 0] <= q[:, n]) & (q[:, n] <= self.accurate_ranges_array[:, 1])
            )
            if is_correct:
                idx_correct.append(n)
            else:
                idx_incorrect.append(n)

        idx_correct = np.array(idx_correct)
        idx_incorrect = np.array(idx_incorrect)

        # split
        def split_consecutive(idx):
            if len(idx) == 0:
                return [], []
            splits = np.split(idx, np.where(np.diff(idx) != 1)[0] + 1)
            all_q = []
            for split in splits:
                mvt = np.zeros((self.model.nb_q, len(split)))
                for k in range(len(split)):
                    mvt[:, k] = q[:, k]
                all_q.append(mvt)
            return splits, all_q

        all_correct_idx, all_correct_q = split_consecutive(idx_correct)
        all_incorrect_idx, all_incorrect_q = split_consecutive(idx_incorrect)

        return all_correct_idx, all_incorrect_idx, all_correct_q, all_incorrect_q


def plot_ranges_with_true_button(ranges_by_joint, accurate_ranges, path_to_save="", show_plot=True):

    n_q = len(ranges_by_joint)
    nb_line, nb_column = n_q, 1
    fig = make_subplots(
        rows=nb_line,
        cols=nb_column,
        subplot_titles=[f"q{idx} - {q_name}" for idx, q_name in enumerate(ranges_by_joint)],
    )

    legend_added = {"pos": False, "neg": False, "zero": False}
    trace_indices_true = []

    for idx_q, q_name in enumerate(ranges_by_joint):
        row, col = idx_q + 1, 1
        muscles = list(ranges_by_joint[q_name].keys())
        component_names = list(ranges_by_joint[q_name].keys())

        for m_name in muscles:
            all_ranges = ranges_by_joint[q_name][m_name]
            true_ranges = accurate_ranges.get(q_name, {}).get(m_name, [])

            for r in all_ranges:
                a, b = r["range"]
                sign = r["sign"]
                width = b - a

                # marquer si cette trace est "vraie"
                if accurate_ranges != {}:

                    is_true = any(tr["range"] == r["range"] and tr["sign"] == r["sign"] for tr in true_ranges)
                    trace_indices_true.append(is_true)
                else:
                    is_true = True

                # couleur et pattern par défaut
                if sign > 0:
                    color = "rgba(214,39,40,0.6)"
                    pattern = "+"
                    legend_key = "pos"
                    name = "Agonist region (+)"
                elif sign < 0:
                    color = "rgba(44,160,44,0.6)"
                    pattern = "-"
                    legend_key = "neg"
                    name = "Antagonist region (-)"
                else:
                    color = "rgba(200,200,200,0.6)"
                    pattern = ""
                    legend_key = "zero"
                    name = "Zero"

                show_legend = is_true and not legend_added[legend_key]
                if show_legend:
                    legend_added[legend_key] = True

                # trace normale
                trace = go.Bar(
                    x=[width],
                    y=[m_name],
                    base=a,
                    orientation="h",
                    marker=dict(
                        color=color,
                        line=dict(color="black", width=1),
                        pattern=dict(shape=pattern),
                    ),
                    name=name,
                    showlegend=show_legend,
                )
                fig.add_trace(trace, row=row, col=col)

        fig.update_xaxes(title_text="Range (rad)", row=row, col=col)

        fig.update_yaxes(
            categoryorder="array",
            categoryarray=component_names[::-1],
            tickmode="array",
            tickvals=component_names,
            ticktext=component_names,
            row=row,
            col=col,
        )

    # bouton pour masquer les traces "fausses"
    buttons = [
        dict(
            label="All ROM",
            method="update",
            args=[
                {"marker.color": [fig.data[i].marker.color for i in range(len(fig.data))]},
                {"title": "All ROM"},
            ],
        ),
    ]

    if accurate_ranges != {}:
        buttons.append(
            dict(
                label="True ROM",
                method="update",
                args=[
                    {
                        "marker.color": [
                            (fig.data[i].marker.color if trace_indices_true[i] else "black")
                            for i in range(len(fig.data))
                        ]
                    },
                    {"title": "True ROM"},
                ],
            ),
        )
    title = f"sign_moment_arm_{path_to_save.replace('/', '_')}"
    fig.update_layout(
        title=title,
        barmode="overlay",
        updatemenus=[dict(type="buttons", showactive=True, buttons=buttons)],
        height=300 * nb_line,
        legend=dict(
            x=1.02,
            y=1,
            xanchor="left",
            yanchor="top",
            orientation="v",
            traceorder="normal",
        ),
    )
    if path_to_save:
        fig.write_html(f"{path_to_save}/{title}.html")
    if show_plot:
        fig.show()


def plot_q_qdot_rom(
    model,
    t,
    q,
    bounds,
    all_correct_idx,
    all_incorrect_idx,
    path_to_save="",
    show_plot=True,
):

    nb_dof = q.shape[0]

    fig = make_subplots(
        rows=nb_dof,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=model.dof_names,
    )

    for idx_dof in range(nb_dof):
        row = idx_dof + 1

        # -------- segments corrects --------
        for k, seg in enumerate(all_correct_idx):
            fig.add_trace(
                go.Scatter(
                    x=t[seg],
                    y=q[idx_dof, seg],
                    mode="lines",
                    line=dict(color="blue"),
                    name="Usable moment range",
                    showlegend=(idx_dof == 0 and k == 0),
                ),
                row=row,
                col=1,
            )

        # -------- segments incorrects --------
        for k, seg in enumerate(all_incorrect_idx):
            fig.add_trace(
                go.Scatter(
                    x=t[seg],
                    y=q[idx_dof, seg],
                    mode="lines",
                    line=dict(color="red", dash="dashdot"),
                    name="Non usable moment range",
                    showlegend=(idx_dof == 0 and k == 0),
                ),
                row=row,
                col=1,
            )

        # -------- bounds --------
        for j in range(2):
            fig.add_trace(
                go.Scatter(
                    x=t,
                    y=[bounds[idx_dof, j]] * len(t),
                    mode="lines",
                    line=dict(color="black", dash="dot"),
                    name="Moment-consistent limits",
                    showlegend=(idx_dof == 0 and j == 0),
                ),
                row=row,
                col=1,
            )

        fig.update_yaxes(title_text="q (rad)", row=row, col=1)

    # -------- lignes verticales de transition --------
    transition_indices = []

    for seg_list in [all_correct_idx, all_incorrect_idx]:
        for seg in seg_list:
            transition_indices.append(seg[0])
            transition_indices.append(seg[-1])

    for idx in np.unique(transition_indices):
        fig.add_vline(
            x=t[idx],
            line=dict(color="gray", dash="dash"),
        )

    # -------- affichage axe X uniquement en bas --------
    for i in range(1, nb_dof):
        fig.update_xaxes(showticklabels=False, row=i, col=1)

    fig.update_xaxes(title_text="Time (s)", row=nb_dof, col=1)

    fig.update_layout(
        height=300 * nb_dof,
        title=f"Joint states and ROM limits\n{path_to_save.replace('/', '_')}",
        template="plotly_white",
        legend=dict(
            x=1.02,
            y=1,
            xanchor="left",
            yanchor="top",
        ),
    )
    if path_to_save:
        fig.write_html(f"{path_to_save}/Joint_states_and_ROM_limits.html")
    if show_plot:
        fig.show()


if __name__ == "__main__":
    # 1) Create a moment arm analyzer for a model
    # --------------------------------------------
    # Path to the model to analyze
    model_path = "models/arm26_allbiceps_1dof.bioMod"
    path_to_save = "examples/data"

    # Load the model
    model = BiomechanicalModelReal().from_biomod(model_path)

    # Create the moment arm analyzer
    moment_arm_analyser = MuscleMomentArmAnalyzer(model_path)

    # You can now access all moment arm sign ranges for each DOF/muscle
    # Each range corresponds to either a negative or positive sign
    # If the sign is 0, the moment arm is null over the entire ROM
    print(moment_arm_analyser.ranges_by_joint)

    # Get the ranges for a specific DOF and muscle
    print(moment_arm_analyser.ranges_by_joint[model.dof_names[0]][model.muscle_names[0]])

    # Alternative way using indices
    print(moment_arm_analyser.get_ranges_from_idx_q_and_m(moment_arm_analyser.ranges_by_joint, 0, 0))

    # Get all muscle ranges for one DOF
    print(moment_arm_analyser.ranges_by_joint[model.dof_names[0]])

    # Alternative way using DOF index only
    print(moment_arm_analyser.get_ranges_from_idx_q(moment_arm_analyser.ranges_by_joint, 0))

    # Visualize the computed ranges
    plot_ranges_with_true_button(
        moment_arm_analyser.ranges_by_joint,
        moment_arm_analyser.accurate_ranges_by_joint,
        path_to_save="data",
    )

    # 2) Specify the expected sign of the moment arms
    # ------------------------------------------------

    # You can define the expected moment arm sign in two ways:

    # a) Using a NumPy array:
    # One row per DOF, one column per muscle
    sign_lever_arm_user = np.array(
        [
            [-1, -1],
        ]
    )

    # Validate and update the expected signs
    moment_arm_analyser.create_sign_lever_arm_user(sign_lever_arm_user)

    # b) Using an explicit dictionary:
    # (Note: this reflects the internal data structure)

    # Define the expected sign
    sign_lever_arm_user = {
        "r_ulna_radius_hand_rotation1_rotZ": {
            "BIClong": -1,
            "BICshort": -1,
        },
    }

    # Update the expected signs
    moment_arm_analyser.update_sign_lever_arm(sign_lever_arm_user)

    # Filter the ranges based on the expected sign
    # and keep only the consistent ranges
    moment_arm_analyser.accurate_ranges_from_true_sign()
    print(moment_arm_analyser.accurate_ranges_by_joint)

    # You can also use get_ranges_from_idx_q_and_m() and get_ranges_from_idx_q()

    # Generate the final ROM array from the filtered ranges
    accurate_ranges = moment_arm_analyser.create_accurate_rom()
    print(accurate_ranges)

    # Visualize the filtered ranges
    plot_ranges_with_true_button(
        moment_arm_analyser.ranges_by_joint,
        moment_arm_analyser.accurate_ranges_by_joint,
        path_to_save="data",
    )

    # 3) Extract usable q(t)
    # -----------------------

    # Provide q(t) and retrieve only the valid (usable) segments

    # Create an arbitrary movement
    N = 100
    q = np.zeros((model.nb_q, N))
    for idx_q in range(model.nb_q):
        q[idx_q, :] = np.linspace(0.0, np.pi, N)

    # Get indices and values of valid and invalid portions of q(t)
    all_correct_idx, all_incorrect_idx, all_correct_q, all_incorrect_q = moment_arm_analyser.get_correct_part_mvt(q)

    # Visualize q(t) together with ROM and valid/invalid segments
    plot_q_qdot_rom(
        model,
        np.linspace(0.0, 1, N),
        q,
        moment_arm_analyser.accurate_ranges_array,
        all_correct_idx,
        all_incorrect_idx,
        path_to_save="data",
    )
