Installation
============

This repository requires Python 3.9 and Pytorch 2.1 or greater. To install the latest version run:

::
    
    pip install flow-matching

Development
-----------------

To create a conda environment with all required dependencies, run:

::
    
    conda env create -f environment.yml
    conda activate flow_matching

Install pre-commit hook. This will ensure that all linting is done on each commit

::
    
    pre-commit install
    conda activate flow_matching


Install the `flow_matching` package in an editable mode:

::
    
    pip install -e .

