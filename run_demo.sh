#!/usr/bin/env bash
mkdir -p ./out_demo
cd ./src
python ./main.py ../demo_dataset ../out_demo -vvvvv --optimize --select-features --eval 2 --feature-reduction 3 --segment --reanalyse
