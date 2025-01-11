#!/bin/sh
python3 tsne_sttm.py --model_name $1 --dataset $2 --dtype $3 --num_topics $4 --num_runs $5
