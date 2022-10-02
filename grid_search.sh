#!/usr/bin/env bash

for lr in 0.1 0.05 0.01 0.005; do 
    for weight_decay in 0.0001 0.0005 ; do
        for batch_size in 32 64 128 256 512; do
            for n_hidden in 64 128 256 512; do
                for dropout in 0.5 0.6 0.7 0.8; do
                    python train.py -b $batch_size -lr $lr -wd $weight_decay --n-hidden $n_hidden -dp $dropout;
                done;
            done;
        done;
    done; 
done