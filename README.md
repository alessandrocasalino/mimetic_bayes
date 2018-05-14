# mimetic_bayes

Simple Bayesian inference software for mimetic gravity

Note on installation and compilation (on Mac):
- install gcc-8 with Homebrew
- compile using gcc-8 -O2 mimetic_evolve.c -o mimetic_evolve.exe

This is necessary due to some incompatibilities with inline functions on the default Mac compiler (clang), and for OpenMP to work properly.
