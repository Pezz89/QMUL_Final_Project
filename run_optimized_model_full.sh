#!/usr/bin/env bash
mkdir -p ./out_full
cd ./src
python ./main.py ../PCG_Physionet_Dataset ../out_full -vvvvv --optimize --select-features 50 --eval 1000 --reanalyse --segment
