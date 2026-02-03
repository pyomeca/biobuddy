from pathlib import Path

from biobuddy import BiomechanicalModelReal, MuscleValidator


def write_graphviz(
    model,
    out_path: str,
    ghost_segments: bool = True,
    via_points: bool = True,
    markers: bool = True,
):
    """
    Write a graphviz .dot file representing the biomechanical model structure

    input:
    segments: dict
    musclegroups: dict
    muscles: dict
    viapoints: dict
    out_path: str, path to save the .dot file
    """
    with open(out_path, "w") as f:
        f.write("digraph biomech {\n")
        f.write("  rankdir=TB;\n")
        f.write('  node [fontname="Helvetica"];\n\n')

        # -------- SEGMENTS --------
        f.write("  // Segments\n")
        for s in model.segments:
            if s.inertia_parameters:
                f.write(
                    f'  "{s.name}" [shape=box, style="filled,bold", fillcolor=lightgray];\n'
                )
            elif ghost_segments:
                f.write(f'  "{s.name}" [shape=box, style=rounded];\n')

            else:
                continue

        f.write("\n  // Segment hierarchy\n")

        # if ghost_segments:
        for s_name in model.segment_names:
            for children in model.children_segment_names(s_name):
                f.write(f'  "{s_name}" -> "{children}";\n')
        # else :

        # for s in model.segments:
        #     if s.internia_parameters :
        #         for children in model.children_segment_names(s_name):

        #         if ghost_segments or (
        #             model.segments[s_name].inertia_parameters
        #             and model.segments[children].inertia_parameters
        #         ):
        #             f.write(f'  "{s_name}" -> "{children}";\n')

        # -------- MARKERS --------
        if markers:
            f.write("\n  // Markers\n")
            for marker_name in model.marker_names:
                f.write(
                    f'  "{marker_name}" [shape=octagon, style=filled, fillcolor=lightgreen];\n'
                )

            for s in model.segments:
                for marker in s.markers:
                    f.write(f'  "{s.name}" -> "{marker.name}";\n')

        # # -------- MUSCLE GROUPS --------
        f.write("\n  // Muscle groups\n")
        for mg in model.muscle_groups:
            origin = mg.origin_parent_name
            insertion = mg.insertion_parent_name

            label = f"{mg.name}\\n({origin} → {insertion})"
            f.write(
                f'  "{mg.name}" [shape=parallelogram, style=filled, fillcolor=lightblue, label="{label}"];\n'
            )
            # origin ---> insertion
            if origin:
                f.write(f'  "{origin}" -> "{mg.name}" [label="origin"];\n')
            if insertion:
                f.write(f'  "{mg.name}" -> "{insertion}" [label="insertion"];\n')

        # # -------- MUSCLES --------
        f.write("\n  // Muscles\n")
        for mg in model.muscle_groups:
            for m_name in mg.muscles.keys():
                f.write(
                    f'  "{m_name}" [shape=diamond, style=filled, fillcolor=lightcoral];\n'
                )
                f.write(f'  "{mg.name}" -> "{m_name}";\n')

        if via_points:
            # # -------- VIAPOINTS --------
            f.write("\n  // Via points\n")
            for vp_name in model.via_point_names:
                f.write(
                    f'  "{vp_name}" [shape=ellipse, style=filled, fillcolor=orange];\n'
                )

            # # -------- MUSCLE --> VIAPOINT CHAINS --------
            f.write("\n  // Muscle paths\n")

            for mg in model.muscle_groups:
                for m in mg.muscles:
                    if not m.via_points:
                        continue
                    # muscle -> first VP
                    f.write(
                        f'  "{m.name}" -> "{m.via_points[0].name}" [label="via"];\n'
                    )

                    # other VP
                    for i in range(len(m.via_points) - 1):
                        f.write(
                            f'  "{m.via_points[i].name}" -> "{m.via_points[i+1].name}" [label="via"];\n'
                        )

                # # last VP -> parent segment
                # last_vp = vps[-1]
                # if last_vp["parent"]:
                #     f.write(
                #         f'  "{last_vp["name"]}" -> "{last_vp["parent"]}" [label="attach"];\n'
                #     )

        f.write("\n}\n")


def create_graph_from_bioMod(
    model,
    base_name: str,
    folder: str = "",
    ghost_segments: bool = True,
    via_points: bool = True,
    markers: bool = True,
):

    path = os.path.join(folder, base_name)

    write_graphviz(model, f"{path}.dot", ghost_segments, via_points, markers)

    dot_file = f"{path}.dot"
    png_file = f"{path}.png"

    subprocess.run(["dot", "-Tpng", dot_file, "-o", png_file])
    print(f"A graph have been created for {base_name} here : {path}.png")


import os
import subprocess

# Get the model
biomod_path = "examples/models/arm26.bioMod"
base_name = "arm26"
model = BiomechanicalModelReal().from_biomod(biomod_path)
create_graph_from_bioMod(
    model, base_name, ghost_segments=True, via_points=True, markers=True
)


print("")
