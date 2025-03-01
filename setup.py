from setuptools import setup


def read_requirements():
    with open("requirements.txt") as f:
        return f.read().splitlines()


setup(install_requires=read_requirements())
