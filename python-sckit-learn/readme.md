# Introduction to Machine Learning with Scikit-Learn

**Code &amp; Data for Introduction to Machine Learning with Scikit-Learn**

[![Scikit-Learn Cheat Sheet](docs/img/cheat_sheet.png)](http://scikit-learn.org/stable/tutorial/machine_learning_map/)

## Installing Scikit-Learn with pip

See the full [installation instructions](http://scikit-learn.org/stable/install.html) for more details; these are provided for convenience only.

Scikit-Learn requires:

- Python >= 2.6 or >= 3.3
- Numpy >= 1.6.1
- SciPy >= 0.9

Once you have installed `pip` (the python package manager):

### Mac OS X

This should be super easy:

    pip install -U numpy scipy scikit-learn

Now just wait! Also, you have no excuse not to do this in a virtualenv.

### Windows

Install [numpy](http://numpy.scipy.org/) and [scipy](http://www.scipy.org/) with their official installers. You can then use PyPi to install scikit-learn:

    pip install -U scikit-learn

If you're having trouble, consider one of the unofficial windows installers or anacondas (see the Scikit-Learn page for more).

### Ubuntu Linux

Unfortunately there are no official binary packages for Linux. First install the build dependencies:

    sudo apt-get install build-essential python-dev python-setuptools \
        python-numpy python-scipy \
        libatlas-dev libatlas3gf-base

Then you can build (hopefully) Scikit-learn with pip:

    pip install --user --install-option="--prefix=" -U scikit-learn

Keep in mind however, that there are other dependencies and might be issues with ATLAS and BLAS - see the official installation for more.
