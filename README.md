<img src="./docs/image/PV_Cell_Logo.png" width="350" class="center">


PVC is a group of four who dream of contributing to the clean energy technology. We developed a python package that can predict **Power Conversion Efficiency(PCE)** of an organic material in PV-Cell based on user's input molecular structure. The predicted model is built based on correlations between *PCE* and molecular features (bond type, functional group, heteroatom and etc.). All data is retrieved from The Harvard Clean Energy Project Database (HCEPDB). As our slogan said "STRONG!", we aimed at developing a powerful tool that provides practical information towards synthesizing new organic materials for OPVC.

## Use Cases
1. Extract molecular features (functional groups, chemical bonding and etc.) from given ``SMILE_str``.
2. Use various regression models to screen siginificant predictors from molecular features above that contribute to *PCE*.
3. Build up Artificial Neural Network(ANN) to connect selected molecular featrues and *PCE*.
4. (Future Work)Predict optimal structure that gives high PCE based on Terminal-Spacer-Core fragmented structures.

## Package Requirements
This package needs **RDkit** for molecular conversion and **Mordred** for descriptor calculation, **Pandas** for data management, **Scikit-learn** for standardisation and data set splitting, and **Keras** for the neural network building and training.

* RDkit
* Keras
* Scikit-learn
* Mordred

All required software can be installed at the command line
 * `pip install -r requirements.txt`
 
## Organization of the  project

The project has the following structure:

    PV_Cell/
      |- README.md
      |- pv_cell/
         |- __init__.py
         |- pv_cell.py
         |- due.py
         |- data/
            |- ...
         |- tests/
            |- ...
      |- doc/
         |- Makefile
         |- conf.py
         |- sphinxext/
            |- ...
         |- _static/
            |- ...
            |- ipynb/
      |- examples/
         |-pv_cell.ipynb/
      |- setup.py
      |- .travis.yml
      |- .mailmap
      |- appveyor.yml
      |- LICENSE
      |- Makefile


## Module code

We place all the module codes in the directory called `pv_cell`. It contains modules
required for every step in the project, including: extracting chemical infomation, 
vaious regression model, ANN model and visualization module. Please see README file in 
the directory for more details. 


## Project Data

All the data used in this project are placed in the directory `database/`
(https://github.com/sjluozho/PV_Cell/tree/master/Database)recorded in csv
files. Users can call the data by this code:
    *data = pd.read_csv('../Database/HCEPD_100K.csv')*
In addition to the original data (*HCEPD_100K.csv*), we place some csv files that
contains raw chemical features extracted from this 100K data, and also the ready-to-use one
in `No_Missing_Value` folder.    


## Examples

This directory contains several Ipython Notebook that reads in some data, and run the modules in
this project as an demonstration. Users may find useful instruction and illustration of the codes
and procedures of our work.


## Installation

For installation and distribution we will use the python standard
library `distutils` module. This module uses a `setup.py` file to
figure out how to install your software on a particular system. For a
small project such as this one, managing installation of the software
modules and the data is rather simple.

A `PV_Cell/version.py` contains all of the information needed for the
installation and for setting up the [PyPI
page](https://pypi.python.org/pypi/PV_Cell) for the software. This
also makes it possible to install your software with using `pip` and
`easy_install`, which are package managers for Python software. The
`setup.py` file reads this information from there and passes it to the
`setup` function which takes care of the rest.

Much more information on packaging Python software can be found in the
[Hitchhiker's guide to
packaging](https://the-hitchhikers-guide-to-packaging.readthedocs.org).


## Licensing

For more details on the licensing document, please read:
[MIT](https://github.com/sjluozho/PV_Cell/blob/master/LICENSE)
