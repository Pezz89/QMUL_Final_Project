#!/usr/bin/env bash

args=("$@")

GCC=${args[0]}

pip install numpy
pip install -r ./src/requirements.txt
marg="try; mex -v GCC='$GCC' viterbi_Springer.c; catch; end; quit"
echo "Running: matlab -nodesktop -nosplash -r $marg"
cd ./src/SpringerExtraction/
matlab -nodesktop -nosplash -nojvm -r "$marg"
matlab -nodesktop -nosplash -nojvm -r "try; a=challenge('../../demo_dataset/training-a/a0001'); if a==-1; disp('MEX completed successfully'); else; disp('MEX failed'); end; catch; end; quit"
