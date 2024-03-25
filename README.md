# SADIC v2: A Modern Implementation of the Simple Atom Depth Index Calculator

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![PyPI version](https://badge.fury.io/py/sadic.svg)](https://badge.fury.io/py/sadic)
[![Downloads](https://pepy.tech/badge/sadic)](https://pepy.tech/project/sadic)

This repository contains the source code for the SADIC v2 package, a modern implementation of the Simple Atom Depth Index Calculator, used to compute the SADIC depth index, a measure of atom depth in protein molecules.

The package is designed to be easy to use and to provide a fast and efficient implementation of the algorithm.

It is built to be used as a command line tool or as a Python library.
The package exposes functions to all the single steps of the algorithm, allowing the user to have full control over the computation, as well as a high-level function to compute the SADIC indices with a single line of python code.

**Authors**
- Giacomo Nunziati
- Alessia Lucia Prete
- Sara Marziali

<!-- ### Table of Contents
- **[Install](#install)**
  - **[Install from PyPI](#install-from-pypi)**
  - **[Install from source](#install-from-source)**
  - **[Requirements](#requirements)**
- **[Usage](#usage)**
  - **[Command Line Interface (CLI)](#command-line-interface-cli)**
  - **[Python interface](#python-interface)**
    - **[Simple usage](#simple-usage)**
    - **[Filter and aggregations](#filter-and-aggregations)**
- **[Software](#software)**
  - **[Main algorithm](#main-algorithm)**
  - **[Architecture](#architecture)**
  - **[Functionalities](#functionalities)**
<!-- - **[Citing](#citing)**
- **[License](#license)** -->

## Install
### Install from PyPI
```bash
  pip install sadic
```

### Install from source
```bash
  git clone https://github.com/nunziati/sadic.git
  cd sadic
  pip install .
```

### Requirements
SADIC v2 requires the packages **numpy**, **scipy**, **biopandas** and **biopython**.

The requirements are automatically downloaded while installing this package.

To in stall the requirements separately, run the following command:

```bash
  pip install -U -r requirements.txt
```

## Usage
The algorithm processes a protein structure and computes the depth index for each atom in the structure.  
The protein structure can be provided as a PDB code or as a path to a PDB file.
The package is integrated with BioPython and BioPandas, so the input can also be provided as a BioPython Structure object or a BioPandas PDB Entity object.

### Command Line Interface (CLI)
Simplified interface for the command line usage of the package.  
The CLI interface only allows to specify the input as a PDB code or a path to a PDB file. The output is returned as a PDB file.
```bash
  sadic <input> --output <output> [--config <config_file>]
```

Input can be:
- a PDB code of a protein structure
- a path to a PDB file (.pdb or .tar.gz)

Output must be a path of a PDB file (.pdb or .tar.gz)

Config file is optional and, if specified, must be a path to a python file (.py) containing two dictionaries:
- sadic_config: a dictionary containing the configuration parameters for the SADIC algorithm
- output_config: a dictionary containing the configuration parameters for the output file

---

### Python interface
#### Simple usage
```python
  import sadic

  # Input protein
  pdb_code = "1GWD" 

  # Run the pipeline
  result = sadic.sadic(pdb_code)

  # (optional) Useful to retrieve the depth indices from the result object
  output = result.get_depth_index()

  # Save the output to a file
  result.save_pdb("1gwd_sadic.pdb")
```

#### Filter and aggregations
Note: filters, atom aggregations and model aggregations are optional and independent from each other.<br>
They can be used in any combination.
```python
  import sadic

  # Input protein
  pdb_code = "1GWD" 

  # Define the filter options
  # Only return the SADIC indices for the atoms composing the alanine and glycine residues
  filter_arg = {"residue_name": ["ALA", "GLY"]}

  # Define the atom aggregation options
  # Compute the depth index for each residue by averaging the depth indices of the atoms composing it
  group_by = "residue_number"
  aggregation_function = "mean"
  atom_aggregation_arg = (group_by, aggregation_function)

  # Define the model aggregation options
  # If the pdb file contains multiple models, they can be aggregated
  # In this case, the depth indices of corresponding atoms in different models are averaged
  model_aggregation_arg = "mean"

  # Run the pipeline
  # Filter by residue name
  result = sadic.sadic(pdb_code, filter_by = filter_arg)

  # (optional) Useful to retrieve the depth indices from the result object
  # Aggregate the depth indices of the atoms of the same residue
  output = result.get_depth_index(atom_aggregation = atom_aggregation_arg)

  # Save the output to a file
  # Aggregate the depth indices of the different models
  result.save_pdb("1gwd_sadic.pdb", model_aggregation = model_aggregation_arg)
```

## Software
Our approach involves modeling each protein as a solid object composed of spheres centered on single atoms.  
SADIC simulates the probing of the protein computing the largest sphere inscribed in its molecular structure.  
Let $r$ be the radius of such sphere and $V_{r_{max}}$ its volume.  
During the simulation, the reference sphere is iteratively centered on each atom $i$, and the exposed volume $V_{r,i}$ is calculated.  
The evaluation of the atom depth index $D_{i,r}$ for the $i$-th atom is determined by the formula:
$$ D_{i,r} = \frac{2V_{r,i}}{V_{r_{max}}} $$
The exposed volume $V_{r,i}$ indicates the volume of the portion of the reference sphere centered on the $i$-th atom that does not intersect the solid representation of the protein.

### Main algorithm
#### The execution of the SADIC v2 algorithm is articulated in multiple stages:
- Loading of protein data;
- Creation of the structured PDB entity;
- For each model found in the PDB file:
  - Creation of the continuous-space model of the protein under analysis;
  - Voxelization and definition of the discrete-space model approximating the protein solid;
  - Filling of the internal cavities of the protein;
  - Computation of the reference radius, that will be used for the depth index calculation;
  - Computation of the depth indexes for the atoms selected by the user.

<!-- ![Main SADIC algorithm](path/to/image.png) -->

<!-- For a more comprehensive understanding of the implementation and analysis of the SADIC v2 algorithm's main pipeline, additional details are available in WAITING FOR CITATION, that presents a deeper exploration and analysis of the algorithm's execution stages. -->

---

### Architecture 
The software architecture of SADIC v2 is organized into distinct sub-packages:
- **pdb** for organizing the data of the input protein and managing the result of the execution of the algorithms;
- **solid** for modeling and manipulating the continuous-space and discrete-space solids representing the molecule;
- **algorithm** where the core algorithms are defined

The main **sadic** package exposes an API with a single function for executing the depth index computation pipeline.

<!-- ![SADIC v2 package architecture](path/to/image.png) -->

---

### Functionalities
#### Different types of input are supported:
- PDB code
- PDB file (raw *.pdb* or compressed *.tar.gz*)
- BioPython Structure object
- BioPandas PDB Entity object


#### The user can specify different options:
- Reference sphere radius
- Van Der Waals radii for the atoms
- Grid resolution for the discretization of the protein
- Protein models to consider (in case of multiple models)
- Atom filters, to select only a subset of atoms
- Atom aggregations, to compute the depth index for groups of atoms
- Model aggregations, to obtain a single depth index for each atom (in case of multiple models)

#### The output can be obtained in different forms:
- Python list
- Numpy array
- Save to a .txt file
- Save to a .npy file (NumPy)
- PDB file (raw *.pdb* or compressed *.tar.gz*)

<!-- ## Citing
**_Waiting for submission/preprint/pubblication_** -->

## License
This project is [MIT](https://choosealicense.com/licenses/mit/) licensed.