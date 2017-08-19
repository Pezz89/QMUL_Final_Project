### Getting things set up.

## Prerequisite

It is assumed that the following libraries, packages and programs are installed
prior to following these instructions. These will most likely be already
installed or can be easily installed via a package manager such as apt-get or
homebrew:

- MATLAB 2017a (it is also assumed that the 'matlab' command has been symlinked
to the user's path and is accessible from the commandline)
- libsndfile                    ([sudo] apt-get install libsndfile1-dev | brew install libsndfile)
- GCC >= 4.9.X                  ([sudo] apt-get install gcc-4.9 | brew install gcc)
- Python 2.7.11                 ([sudo] apt-get install python | brew install python)
- Pip (Python's package manager - not always included with python distribution)

(It is likely that code will run on other versions than those stated, however
this is not guaranteed)


## Installation Instructions

This installation guide was written using a clean install of Ubuntu 17.04,
running on virtualbox 5.1.26.

Providing all prerequisite packages are installed correctly, running:

./install.sh /path/to/GCC-4.9

from the project directory should install all python dependencies
automatically. An absolute path to a version of GCC compatible with Matlab's
MEX command is required for compilation of segmentation scripts

If this is successful, the program should exit, stating the MEX command ran
successfully.


