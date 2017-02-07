### Getting things set up.
Compiling the RR extraction matlab code:

tested using Matlab 2014b
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
