
![biobuddy](https://github.com/user-attachments/assets/c8689155-0b26-4e13-835c-cdb6696e1acb)

`BioBuddy` is an open-source tool for translating and personalizing musculoskeletal models across different formats (e.g., .osim, .bioMod). By enabling reliable interoperability between modeling environments, BioBuddy allows researchers to focus on scientific questions rather than technical constraints.

# How to install 
Currently, the only way to install `BioBuddy` is from source. But it will be available on conda-forge and pip in the near future.

If you are a user, you can setup your environement however you'd like since we do not have any mandatory dependency.

However, if you are a developper and wuant to contribute, you will need to setup your environment using the following command:
```bash
conda install -c conda-forge python>=3.10 pip 
conda install -c opensim-org opensim
conda install -c conda-forge biorbd pyorerun
pip install pytest pytest-cov codecov black scipy
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
