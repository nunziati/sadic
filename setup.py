r"""setup.py"""

import os

from setuptools import setup


def read(fname):
    r"""Reads a file and returns its content as a string."""
    return open(os.path.join(os.path.dirname(__file__), fname), encoding="utf-8").read()


def get_version(rel_path):
    r"""Gets the version of the package from the __init__ file."""
    for line in read(rel_path).splitlines():
        if line.startswith("__version__"):
            delim = '"' if '"' in line else "'"
            return line.split(delim)[1]

    raise RuntimeError("Unable to find version string.")


setup(
    name="sadic",
    version=get_version("sadic/__init__.py"),
    description="Reimplementation as a python package of the software for Simple Atom Depth Index Calculator (SADIC)",
    url="https://github.com/nunziati/sadic",
    author="Giacomo Nunziati",
    author_email="giacomo.nunziati.0@gmail.com",
    license="MIT License",
    keywords="protein atom depth",
    long_description=read("README.md"),
    long_description_content_type="text/markdown",
    entry_points={
        'console_scripts': [
            'sadic = sadic.cli_sadic:main',
        ],
    },
    packages=[
        "sadic",
        "sadic.algorithm",
        "sadic.quantizer",
        "sadic.utils",
        "sadic.solid",
        "sadic.pdb",
    ],
    install_requires=["numpy", "scipy", "biopandas", "biopython", "tqdm", "matplotlib"],
    classifiers=[
        "Development Status :: 1 - Planning",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Operating System :: POSIX :: Linux",
        "Operating System :: Microsoft :: Windows",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
    ],
)
