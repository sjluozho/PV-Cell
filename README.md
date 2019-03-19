<img src="./docs/image/PV_Cell_Logo.png" width="350" class="center">



PVC is a group of four who dream of contributing to the clean energy technology. We developed a python package that can predict **Power Conversion Efficiency(PCE)** of an organic material in PV-Cell based on user's input molecular structure. The predicted model is built based on correlations between *PCE* and molecular features (bond type, functional group, heteroatom and etc.). All data is retrieved from The Harvard Clean Energy Project Database (HCEPDB). As our slogan said "STRONG!", we aimed at developing a powerful tool that provides practical information towards synthesizing new organic materials for OPVC.

### Use Cases
1. Extract molecular features (functional groups, chemical bonding and etc.) from given ``SMILE_str``.
2. Use various regression models to screen siginificant predictors from molecular features above that contribute to *PCE*.
3. Build up Artificial Neural Network(ANN) to connect selected molecular featrues and *pce*.
4. Predict optimal structure that gives high PCE based on Terminal-Spacer-Core fragmented structures.

### Package Requirements
This package needs RDkit for molecular conversion and descriptor calculation, Pandas for data management, Scikit-learn for standardisation and data set splitting as well as Keras for the neural network building and training.

* RDkit
* Keras
* Scikit-learn
* Mordred

All required software can be installed at the command line
 * `pip install -r requirements.txt`
 
### Organization of the  project

The project has the following structure:

    PV_Cell/
      |- README.md
      |- PV_Cell/
         |- __init__.py
         |- PV_Cell.py
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
      |- setup.py
      |- .travis.yml
      |- .mailmap
      |- appveyor.yml
      |- LICENSE
      |- Makefile
      |- ipynb/
         |- ...


In the following sections we will examine these elements one by one. First,
let's consider the core of the project. This is the code inside of
`PV_Cell/PV_Cell.py`. The code provided in this file is intentionally rather
simple. It implements some simple curve-fitting to data from a psychophysical
experiment. It's not too important to know what it does, but if you are really
interested, you can read all about it
[here](http://arokem.github.io/2014-08-12-learn-optimization.html).


### Module code

We place all the module codes in the directory called `PV_Cell`. It contains modules
required for every step in the project, including: extracting chemical infomation, 
vaious regression model, ANN model and visualization module. Please see README file in 
the directory for more details. 


### Project Data

All the data used in this project are placed in the directory `Database/`
(https://github.com/sjluozho/PV_Cell/tree/master/Database)recorded in csv
files. Users can call the data by this code:
    *data = pd.read_csv('../Database/HCEPD_100K.csv')*
In addition to the original data (*HCEPD_100K.csv*), we place some csv files that
contains raw chemical features extracted from this 100K data, and also the ready-to-use one
in `No_Missing_Value` folder.    


### Testing

Most scientists who write software constantly test their code. That is, if you
are a scientist writing software, I am sure that you have tried to see how well
your code works by running every new function you write, examining the inputs
and the outputs of the function, to see if the code runs properly (without
error), and to see whether the results make sense.

Automated code testing takes this informal practice, makes it formal, and
automates it, so that you can make sure that your code does what it is supposed
to do, even as you go about making changes around it.

Most scientists writing code are not really in a position to write a complete
[specification](http://www.wired.com/2013/01/code-bugs-programming-why-we-need-specs/)
of their software, because when they start writing their code they don't quite
know what they will discover in their data, and these chance discoveries might
affect how the software evolves. Nor do most scientists have the inclination to
write complete specs - scientific code often needs to be good enough to cover
our use-case, and not any possible use-case. Testing the code serves as a way to
provide a reader of the code with very rough specification, in the sense that it
at least specifies certain input/output relationships that will certainly hold
in your code.

We recommend using the ['pytest'](http://pytest.org/latest/) library for
testing. The `py.test` application traverses the directory tree in which it is
issued, looking for files with the names that match the pattern `test_*.py`
(typically, something like our `PV_Cell/tests/test_PV_Cell.py`). Within each
of these files, it looks for functions with names that match the pattern
`test_*`. Typically each function in the module would have a corresponding test
(e.g. `test_transform_data`). This is sometimes called 'unit testing', because
it independently tests each atomic unit in the software. Other tests might run a
more elaborate sequence of functions ('end-to-end testing' if you run through
the entire analysis), and check that particular values in the code evaluate to
the same values over time. This is sometimes called 'regression testing'. We
have one such test in `PV_Cell/tests/test_PV_Cell.py` called
`test_params_regression`. Regressions in the code are often canaries in the coal
mine, telling you that you need to examine changes in your software
dependencies, the platform on which you are running your software, etc.

Test functions should contain assertion statements that check certain relations
in the code. Most typically, they will test for equality between an explicit
calculation of some kind and a return of some function. For example, in the
`test_cumgauss` function, we test that our implmentation of the cumulative
Gaussian function evaluates at the mean minus 1 standard deviation to
approximately (1-0.68)/2, which is the theoretical value this calculation should
have. We recommend using functions from the `numpy.testing` module (which we
import as `npt`) to assert certain relations on arrays and floating point
numbers. This is because `npt` contains functions that are specialized for
handling `numpy` arrays, and they allow to specify the tolerance of the
comparison through the `decimal` key-word argument.

To run the tests on the command line, change your present working directory to
the top-level directory of the repository (e.g. `/Users/arokem/code/PV_Cell`),
and type:

    py.test PV_Cell

This will exercise all of the tests in your code directory. If a test fails, you
will see a message such as:


    PV_Cell/tests/test_PV_Cell.py .F...

    =================================== FAILURES ===================================
    ________________________________ test_cum_gauss ________________________________

      def test_cum_gauss():
          sigma = 1
          mu = 0
          x = np.linspace(-1, 1, 12)
          y = sb.cumgauss(x, mu, sigma)
          # A basic test that the input and output have the same shape:
          npt.assert_equal(y.shape, x.shape)
          # The function evaluated over items symmetrical about mu should be
          # symmetrical relative to 0 and 1:
          npt.assert_equal(y[0], 1 - y[-1])
          # Approximately 68% of the Gaussian distribution is in mu +/- sigma, so
          # the value of the cumulative Gaussian at mu - sigma should be
          # approximately equal to (1 - 0.68/2). Note the low precision!
    >       npt.assert_almost_equal(y[0], (1 - 0.68) / 2, decimal=3)
    E       AssertionError:
    E       Arrays are not almost equal to 3 decimals
    E        ACTUAL: 0.15865525393145707
    E        DESIRED: 0.15999999999999998

    PV_Cell/tests/test_PV_Cell.py:49: AssertionError
    ====================== 1 failed, 4 passed in 0.82 seconds ======================

This indicates to you that a test has failed. In this case, the calculation is
accurate up to 2 decimal places, but not beyond, so the `decimal` key-word
argument needs to be adjusted (or the calculation needs to be made more
accurate).

As your code grows and becomes more complicated, you might develop new features
that interact with your old features in all kinds of unexpected and surprising
ways. As you develop new features of your code, keep running the tests, to make
sure that you haven't broken the old features.  Keep writing new tests for your
new code, and recording these tests in your testing scripts. That way, you can
be confident that even as the software grows, it still keeps doing correctly at
least the few things that are codified in the tests.

We have also provided a `Makefile` that allows you to run the tests with more
verbose and informative output from the top-level directory, by issuing the
following from the command line:

    make test

### Documentation

Documenting your software is a good idea. Not only as a way to communicate to
others about how to use the software, but also as a way of reminding yourself
what the issues are that you faced, and how you dealt with them, in a few
months/years, when you return to look at the code.

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


### Installation

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


### Licensing

For more details on the licensing document, please read:
[MIT](https://github.com/sjluozho/PV_Cell/blob/master/LICENSE)

### Getting cited

When others use your code in their research, they should probably cite you. To
make their life easier, we use [duecredit](http://www.duecredit.org). This is a software
library that allows you to annotate your code with the correct way to cite it.
To enable `duecredit`, we have added a file `due.py` into the main directory.
This file does not need to change at all (though you might want to occasionally
update it from duecredit itself. It's
[here](https://github.com/duecredit/duecredit/blob/master/duecredit/stub.py),
under the name `stub.py`).

In addition, you will want to provide a digital object identifier (DOI) to the
article you want people to cite.

To get a DOI, use the instructions in [this page](https://guides.github.com/activities/citable-code/)

Another way to get your software cited is by writing a paper. There are several
[journals that publish papers about software](https://www.software.ac.uk/resources/guides/which-journals-should-i-publish-my-software).

### Scripts

A scripts directory can be used as a place to experiment with your
module code, and as a place to produce scripts that contain a
narrative structure, demonstrating the use of the code, or producing
scientific results from your code and your data and telling a story
with these elements.

For example, this repository contains an [IPython notebook] that reads
in some data, and creates a figure. Maybe this is *Figure 1* from some
future article? You can see this notebook fully rendered
[here](https://github.com/uwescience/PV_Cell/blob/master/scripts/Figure1.ipynb).

