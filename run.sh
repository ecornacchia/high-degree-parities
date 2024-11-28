#!/bin/bash

for seed in 4 5 6 7 8 9 10
do

    for sigma in 0 0.1 1 -1
    do
        for trainsize in 10000 50000 90000 130000
        do
             
            python3 main.py -task fullparity -model mlp -lr 0.1 -seed $seed  -train-size $trainsize -cuda 0 -eps 0.01 -sigma_init $sigma
            
        done
    done
done
