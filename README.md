# mimetic_bayes

Simple Bayesian inference software for mimetic gravity

Note on installation and compilation (on Mac):
- install gcc-8 with Homebrew
- compile using gcc-8 -O2 -DHAVE_INLINE -DGSL_RANGE_CHECK_OFF Ligo_abc_gsl.c -o Ligo_abc_gsl.exe -lgsl -lgslcblas -fopenmp
- run with ./Ligo_abc_gsl.exe
- (optional) for computational time tests: time ./Ligo_abc_gsl.exe

This is necessary due to some incompatibilities with inline functions on the default Mac compiler (clang), and for OpenMP to work properly.
