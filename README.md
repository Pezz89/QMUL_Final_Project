### Getting things set up.
Compiling the RR extraction matlab code:

tested using Matlab 2017a
try running (from the ./src/RR_Extraction/ folder):

>> mex viterbi_Springer.c

if running on a mac this may produce an error about not being able to find a
compiler. If so, follow instructions by Ken Atwell here: https://uk.mathworks.com/matlabcentral/answers/243868-mex-can-t-find-compiler-after-xcode-7-update-r2015b

Having completed this run the mex command withut exiting matlab and the C file
should compile to a mex file. from here, run:

challenge('../../validation_dataset/a0001')

this should return:

ans =

    -1

## Prerequisite

It is assumed that the following libraries, packages and programs are installed
prior to following these instructions. These will most likely be already
installed or can be easily installed via a package manager such as apt-get or
homebrew:

- MATLAB 2017a (it is also assumed that the 'matlab' command has been symlinked
to the user's path and is accessible from the commandline)
- libsndfile
- Python 2.7.11

(It is likely that code will run on other versions than those stated, however
this is not guaranteed)



## Installation Instructions

This installation guide was written using a clean install of Ubuntu 17.04,
running on virtualbox 5.1.26.



