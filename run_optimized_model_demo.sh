#!/usr/bin/env bash
mkdir -p ./out_optimized_demo
echo "Copying model output directory..."
cp -rf ./prebuilt_models/SavedModel5/out/ ./out_optimized_demo
cd ./src
python ./main.py ../demo_dataset ../out_optimized_demo -vvvvv --select-features 0
