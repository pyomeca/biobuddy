"""
This example shows how you can create a model from Yeadon's inertial model either by:
1. Providing the measurements by hand,
2. Providing the standard measurement text file, or
3. Using the GUI to input the measurements.

Please note that this feature depends on the Yeadon library, installable using `pip install yeadon` or `conda install -c conda-forge yeadon`

REF: Yeadon, M. R. (1990). The simulation of aerial movement—II. A mathematical inertia model of the human body. Journal of biomechanics, 23(1), 67-74.
"""

from pathlib import Path

from biobuddy import launch_yeadon_measurement_editor, YeadonTable, YeadonDensitySet, YeadonMeasures

if __name__ == "__main__":

    # 1. Measures
    yeadon_table_measures = YeadonTable(
        symmetric=True,  # Weather the left and right arms and legs should be symmetrized
        density_set=YeadonDensitySet.DEMPSTER,
        total_mass=81.7,  # Kg
    )
    yeadon_table_measures.from_measurements(
        YeadonMeasures(
            Ls1L=11.9,
            La2L=26.7,
            Lb2L=26.1,
            Ls2L=19.6,
            La3L=36.9,
            Lb3L=34.9,
            Ls3L=34.7,
            La4L=54.1,
            Lb4L=52.3,
            Ls4L=54.0,
            La5L=5.5,
            Lb5L=5.0,
            Ls5L=50.0,
            La6L=8.6,
            Lb6L=7.8,
            Ls6L=12.0,
            La7L=16.0,
            Lb7L=16.1,
            Ls7L=17.1,
            Ls8L=26.3,
            La0p=39.4,
            Lb0p=42.5,
            La1p=34.7,
            Lb1p=33.5,
            Ls0p=93.0,
            La2p=28.2,
            Lb2p=28.2,
            Ls1p=88.0,
            La3p=28.9,
            Lb3p=28.9,
            Ls2p=90.0,
            La4p=17.0,
            Lb4p=17.0,
            Ls3p=101.8,
            La5p=24.5,
            Lb5p=24.5,
            Ls5p=42.1,
            La6p=21.0,
            Lb6p=19.6,
            Ls6p=52.0,
            La7p=11.7,
            Lb7p=11.9,
            Ls7p=57.9,
            La4w=5.6,
            Lb4w=5.7,
            Ls0w=31.1,
            La5w=9.5,
            Lb5w=9.5,
            Ls1w=30.6,
            La6w=7.7,
            Lb6w=7.9,
            Ls2w=31.3,
            La7w=4.8,
            Lb7w=4.6,
            Ls3w=37.1,
            Ls4w=42.5,
            Lj1L=10.1,
            Lk1L=10.1,
            Ls4d=18.2,
            Lj3L=41.0,
            Lk3L=44.0,
            Lj4L=55.4,
            Lk4L=59.7,
            Lj5L=79.7,
            Lk5L=82.9,
            Lj6L=1.0,
            Lk6L=1.0,
            Lj8L=13.5,
            Lk8L=13.2,
            Lj9L=18.0,
            Lk9L=16.4,
            Lj1p=61.5,
            Lk1p=63.5,
            Lj2p=53.8,
            Lk2p=55.6,
            Lj3p=35.5,
            Lk3p=34.5,
            Lj4p=39.0,
            Lk4p=38.3,
            Lj5p=22.9,
            Lk5p=24.7,
            Lj6p=31.3,
            Lk6p=30.9,
            Lj7p=24.6,
            Lk7p=24.3,
            Lj8p=22.6,
            Lk8p=22.9,
            Lj9p=14.1,
            Lk9p=14.6,
            Lj8w=9.2,
            Lk8w=9.1,
            Lj9w=6.2,
            Lk9w=6.5,
            Lj6d=11.8,
            Lk6d=10.7,
        ),
    )

    # 2. Text file
    current_parent_folder = Path(__file__).parent
    file_path = f"{current_parent_folder}/models/yeadon_measures.txt"
    yeadon_table_file = YeadonTable(
        symmetric=True,  # Weather the left and right arms and legs should be symmetrized
        density_set=YeadonDensitySet.DEMPSTER,
    )
    yeadon_table_file.from_file(file_path)

    # 3. GUI
    # Please note that it is possible to fill all the field in the GUI as you take the measurements using:
    # launch_yeadon_measurement_editor()

    # But here we preset the values using the values from the measurement file to illustrate how to modify a table with the GUI.
    launch_yeadon_measurement_editor(yeadon_table_file)

    # 4. Create simple model from this inertia table
    model = yeadon_table_file.to_simple_model()
    model.animate()
