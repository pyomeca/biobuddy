
![biobuddy](https://github.com/user-attachments/assets/c8689155-0b26-4e13-835c-cdb6696e1acb)

`BioBuddy` is an open-source tool for [translating](#model-translation), [creating](#model-creation) and [personalizing](#model-personalization) musculoskeletal models across different formats (e.g., .osim, .bioMod). By enabling reliable interoperability between modeling environments, BioBuddy allows researchers to focus on scientific questions rather than technical constraints.

<!---
[![Actions Status](https://github.com/pyomeca/biobuddy/workflows/CI/badge.svg)](https://github.com/pyomeca/biobuddy/actions)
[![PyPI](https://anaconda.org/conda-forge/biobuddy/badges/latest_release_date.svg)](https://pypi.org/project/biobuddy/)
--->

[![codecov](https://codecov.io/gh/pyomeca/biobuddy/branch/main/graph/badge.svg)](https://codecov.io/gh/pyomeca/biobuddy)
[![Discord](https://img.shields.io/discord/1340640457327247460.svg?label=chat&logo=discord&color=7289DA)](https://discord.gg/Ux7BkdjQFW)

# How to install 
Currently, the only way to install `BioBuddy` is from source. But it will be available on conda-forge and pip in the near future.

If you are a user, you can set up your environment with minimal dependencies.
```bash
conda install -c conda-forge
pip install numpy matplotlib lxml scipy ezc3d
```

However, if you are a developer and want to contribute, you will need to set up your environment using the following command:
Due to the OpenSim dependency used only in BioBuddy's tests, we recommend using python=3.11.
```bash
conda install -c opensim-org opensim
conda install -c conda-forge biorbd pyorerun
pip install numpy matplotlib lxml pytest black scipy ezc3d pyomeca
```

# Model translation
You can load the original model using one of the `BiomechanicalModelReal.from_[format]` methods, and then export it into another format using the `BiomechanicalModelReal.to_[format]` method.
```python3
from biobuddy import BiomechanicalModelReal

# Read an .osim file
model = BiomechanicalModelReal.from_osim(
    filepath=osim_file_path,
    # Other optional parameters here
)

# Translate it into a .bioMod file
model.to_biomod(biomod_file_path)
```

# Model creation
`TODO: complete when the example is ready`

# Model personalization
The current version of BioBuddy allows you to modify your `BiomechanicalModelReal` to personalize it to your subjects.

**Scaling:**
First you need to define the scaling configuration using `ScaleTool(original_model).from_xml(filepath=xml_filepath)`or 
`ScaleTool(original_model).from_biomod(filepath=biomod_filepath)`.
Then, you can use the `scale` method to scale your model. The `scale` method takes the following parameters:
```python3
from biobuddy import ScaleTool

# Scaling configurations
scale_tool = ScaleTool(original_model=original_model).from_xml(filepath=xml_filepath)

# Performing the scaling based on a static trial
scaled_model_CAC = scale_tool.scale(file_path=static_c3d_file_path, first_frame=100, last_frame=200, mass=mass)
```

**Joint center identification:**
The `JointCenterTool` allows you to identify the joint centers of your model based on the movement of segments during functional trials.
First, you need to define the joint center configuration using `JointCenterTool.add()` method to define the joint center you want to relocate.
Then, you can use the `JointCenterTool.perform_tasks()` method to relocate each joint.
```python3
from biobuddy import JointCenterTool

# Set up the joint center identification tool
joint_center_tool = JointCenterTool(scaled_model)
# Example for the right hip
joint_center_tool.add(
    Score(
        file_path=hip_movement_c3d_file_path,
        parent_name="pelvis",
        child_name="femur_r",
        parent_marker_names=["RASIS", "LASIS", "LPSIS", "RPSIS"],
        child_marker_names=["RGT", "RUB_Leg", "RUF_Leg", "FBF_Leg", "RMFE", "RLFE"],
        first_frame=100,
        last_frame=900,
    )
)
# Example for the right knee
joint_center_tool.add(
    Sara(
        file_path=knee_movement_c3d_file_path,
        parent_name="femur_r",
        child_name="tibia_r",
        parent_marker_names=["RGT", "RUB_Leg", "RUF_Leg", "FBF_Leg"],
        child_marker_names=["RATT", "RUB_Tib", "RDF_Tib", "RDB_Tib", "RSPH", "RLM"],
        first_frame=100,
        last_frame=900,
    )
)

# Perform the joint center identification
modified_model = joint_center_tool.perform_tasks()
```

# How to cite
```
@software{biobuddy_2025,
  author       = {Eve Charbonneau, Pierre Puchaud, Teddy Caderby, Mickael Begon, Amedeo Ceglia, Benjamin Michaud},
  title        = {Bringing the musculoskeletal modeling community together with BioBuddy},
  month        = april,
  year         = 2025,
  publisher    = {submitted to Congrès de la Société de biomécanique},
  url          = {https://github.com/pyomeca/biobuddy}
}
```

# How to contribute
Our goal is to support as many musculoskeletal model formats as possible, so do not hesitate to contact us if you'd like to see your favorite format supported by BioBuddy. 
If you are using BioBuddy and encounter any problem, please open an issue on this GitHub repository. 
We are also open to suggestions for new features or improvements to existing functionality.
All contributions are welcome!
