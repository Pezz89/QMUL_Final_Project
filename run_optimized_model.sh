#!/usr/bin/env bash
mkdir -p ./out_optimized
echo "Copying model output directory..."
cp -rf ./prebuilt_models/SavedModel5/out/ ./out_optimized
cd ./src
python ./main.py ../PCG_Physionet_Dataset ../out_optimized -vvvvv --select-features 0
