#! /bin/bash

# test 0
gcc clmul.c -o clmul
./clmul

# test 1
python aes.py

# test 2
torchrun --nproc_per_node=4 crypto_all_reduce.py