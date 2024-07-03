#!/bin/bash
task="wikitable"

#use_sample for sample test, remove use_sample for entire dataset augmentation
python run.py -s test -m Augmentation -t $task --aug_type schema --use_sample 

python run.py -s test -m Augmentation -t $task --aug_type composition summary --use_sample 

# task="tabfact"

# python run.py -s test -m Augmentation -t $task --aug_type schema --use_sample --small_test

# python run.py -s test -m Augmentation -t $task --aug_type composition summary --use_sample --small_test