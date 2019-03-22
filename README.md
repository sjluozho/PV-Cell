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


## Documentation

The first step in this direction is to document every function in your module
code. We recommend following the [numpy docstring
standard](https://github.com/numpy/numpy/blob/master/doc/HOWTO_DOCUMENT.rst.txt),
which specifies in detail the inputs/outputs of every function, and specifies
how to document additional details, such as references to scientific articles,
notes about the mathematics behind the implementation, etc.

This standard also plays well with a system that allows you to create more
comprehensive documentation of your project. Writing such documentation allows
you to provide more elaborate explanations of the decisions you made when you
were developing the software, as well as provide some examples of usage,
explanations of the relevant scientific concepts, and references to the relevant
literature.

To document `PV_Cell` we use the [sphinx documentation
system](http://sphinx-doc.org/). You can follow the instructions on the sphinx
website, and the example [here](http://matplotlib.org/sampledoc/) to set up the
system, but we have also already initialized and commited a skeleton
documentation system in the `docs` directory, that you can build upon.

Sphinx uses a `Makefile` to build different outputs of your documentation. For
example, if you want to generate the HTML rendering of the documentation (web
pages that you can upload to a website to explain the software), you will type:

	make html

This will generate a set of static webpages in the `doc/_build/html`, which you
can then upload to a website of your choice.

Alternatively, [readthedocs.org](https://readthedocs.org) (careful,
*not* readthedocs.**com**) is a service that will run sphinx for you,
and upload the documentation to their website. To use this service,
you will need to register with RTD. After you have done that, you will
need to "import your project" from your github account, through the
RTD web interface. To make things run smoothly, you also will need to
go to the "admin" panel of the project on RTD, and navigate into the
"advanced settings" so that you can tell it that your Python
configuration file is in `doc/conf.py`:

![RTD conf](https://github.com/uwescience/PV_Cell/blob/master/doc/_static/RTD-advanced-conf.png)

 http://PV_Cell.readthedocs.org/en/latest/


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
