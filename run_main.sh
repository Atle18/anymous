#!/bin/bash

python main.py --EM True --dp_sgd True --clip_bound_ista 0.01 --sigma_sgd 0.89 --topk 16 --batch_size 60
