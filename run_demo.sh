#!/usr/bin/env bash
mkdir -p ./out_demo
cd ./src
python ./main.py ../demo_dataset ../out_demo -vvvvv --optimize --select-features 3 --eval 2 --segment --reanalyse
