# mimetic_bayes

Simple Bayesian inference software for mimetic gravity

## Installation and running

Note on installation and compilation (on Mac):
- install gcc-X (where X must be substituted with the current last available version of gcc) with Homebrew
- install gsl library with Homebrew (and link them if necessary)
- compile using gcc-X -O2 -DHAVE_INLINE -DGSL_RANGE_CHECK_OFF Ligo_abc_gsl.c -o Ligo_abc_gsl.exe -lgsl -lgslcblas -fopenmp
- run with ./Ligo_abc_gsl.exe
- (optional) for computational time tests: time ./Ligo_abc_gsl.exe

Use of gcc from Homebrew is necessary to avoid some incompatibilities with inline functions on the default Mac compiler (clang), and for OpenMP to work properly.

## Getdist plot

A very simple getdist plot script is available. It is necessary to install a version of Python, and also getdist and jupyter to run it. Note that these packages are not needed if you only want to use the .c program.

To install these packages install Python with Homebrew, and then open a terminal and write:
- pip install getdist
- pip install jupyter

Note: you might need to use also pip2 or pip3, depending on the Python version you have installed and/or you are using.
