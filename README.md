# Photovoltaic-Cell

## Use Case
1. Extract molecular information (functional groups, chemical bonding, etc.) with given SMILE_str.
2. Predict photovoltaic cell performance of given material by predicting its power conversion efficiency according to its molecular information.

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