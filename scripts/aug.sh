#!/bin/bash

python run.py -s test -m Augmentation -t sqa --aug_type schema --use_sample 

python run.py -s test -m Augmentation -t sqa --aug_type composition --use_sample 