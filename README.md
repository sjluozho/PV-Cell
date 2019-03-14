# OPVCM
Predicting Organic Photovoltaic Cell Material (OPVCM) is a python package that can predict **Power Conversion Efficiency(pce)** of an organic material in PV-Cell based on user's input data. The predicted model is built based on correlations between *pce* and molecular features (bond type, functional group, heteroatom and etc.). All data is retrieved from The Harvard Clean Energy Project Database (HCEPDB).

## Use Cases
1. Extract molecular features (functional groups, chemical bonding and etc.) from given ``SMILE_str``.
2. Use LASSO regression to screen siginificant predictors from molecular features above that contribute *pce*.
3. Build up Neural Network to connect selected molecular featrues and *pce*.

## Package Requirements
This package needs RDkit for molecular conversion and descriptor calculation, Pandas for data management, Scikit-learn for standardisation and data set splitting as well as Keras for the neural network building and training.

* RDkit
* Keras
* Scikit-learn
* Mordred



## Organization of the project
The project has the following structure:

    PV-Cell/
      |- README.md
      |- PV-Cell/
         |- __init__.py
         |- PV-Cell.py
         |- due.py
         |- data/
            |- HCEPD_100K.csv
         |- tests/
            |- test_PV-Cell.py
      |- doc/
         |- Makefile
         |- conf.py
         |- sphinxext/
            |- ...
         |- _static/
            |- ...
      |- setup.py
      |- LICENSE
      |- Makefile
      |- ipynb/
         |- ...
