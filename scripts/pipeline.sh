#!/bin/bash
task="wikitable"

parent_dir=$(dirname "$(pwd)/$(basename "$0")")
mkdir -p "$parent_dir/db/sqlite"
echo "Generating database file for task: $task"
touch "$parent_dir/db/sqlite/$task.db"

# use_sample for sample test, remove use_sample for entire dataset augmentation
python run.py -s test -m Pipeline -t $task --save_file --model gpt-3.5-turbo --use_sample 