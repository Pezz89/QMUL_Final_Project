### Getting things set up.

## Prerequisite

It is assumed that the following libraries, packages and programs are installed
prior to following these instructions. These will most likely be already
installed or can be easily installed via a package manager such as apt-get or
homebrew:

- MATLAB 2017a (it is also assumed that the 'matlab' command has been symlinked
to the user's path and is accessible from the commandline)
- libsndfile                    ([sudo] apt-get install libsndfile1-dev | brew install libsndfile)
- GCC >= 4.9.X                  ([sudo] apt-get install gcc-4.9 | brew install gcc@4.9)
- G++ >= 4.9.X                  ([sudo] apt-get install g++-4.9 | brew install gcc@4.9)
- Python 2.7.11                 ([sudo] apt-get install python | brew install python)
- Pip (Python's package manager - not always included with python distribution)
- multitail                     (Not essential, used for viewing logs)

(It is likely that code will run on other versions than those stated, however
this is not guaranteed)


## Installation Instructions

This installation guide was written using a clean install of Ubuntu 17.04,
running on virtualbox 5.1.26.

Providing all prerequisite packages are installed correctly, running:

[sudo] ./install.sh /path/to/GCC-4.9

from the project directory should install all python dependencies
automatically. An absolute path to a version of GCC compatible with Matlab's
MEX command is required for compilation of segmentation scripts

If this is successful, the program should exit, stating the MEX command ran
successfully.

## Running the program

Scripts and a small demo dataset are provided to demonstrate the operation of the system.
To allow for reasonable running time, the dataset has been reduced
significantly from that used for training/testing of models during development.
Iterations of optimisation algorithms have also been reduced significantly. As a
result, these scripts simply demonstrate that the system run correctly, and are
not a demonstration of it's performance. Performance will be demonstrated in the Viva.

The included scripts are:

run_demo.sh
Complete run of system from start to finish: Trains a model on the
demo_dataset, optimizes for 2 evaluations, picks 3 features and scores
performance using metrics described in the report.

run_optimized_model_demo.sh 
Scoring of properly trained model: An optimized model generated during
development has been included and will be scored on the demo dataset. Again
this is purely a quick demonstration of correct operation, not performance.

run_optimized_model.sh
Scores an optimized model on the full dataset. This should run in a reasonable
amount of time, as features and models are already computed.

run_optimized_model_full.sh
Train a model from scratch on the full dataset. This will take a considerable
amount of time and requires significant computing power. This demonstrates the
method used for development of the final algorithm.

view_logs
Opens log files created when running the program. It is advised that this is
run during particle swarm optimisation, as this is when threaded logging occurs

./src/main.py --help
The underlying interface used for training, optimization and scoring of models.
Running the help flag displays a list of all arguments available to the user.
This can be used for the generation of new models on any dataset (however, it
is highly recommended that the full Physionet dataset is used for best results:
https://physionet.org/physiobank/database/challenge/2016/)
