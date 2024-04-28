import os
import json
import argparse
import numpy as np
from pipeline import pipeline
from batch_pipe import new_pipeline
from augmentation import augmentation
import logging

def add_arguments():
    # Create an ArgumentParser object
    parser = argparse.ArgumentParser(description='Description of your code running.')

    # Add arguments
    parser.add_argument(
        '-t',
        '--task_name',
        default='tabfact',
        # choices=[task.value for task in TaskName],
        type=str,
        help='Task name.',
    )
    parser.add_argument(
        '-s',
        '--split',
        default="validation",
        choices=["train", "test", "validation"],
        type=str,
        help='Split of the dataset.',
    )
    parser.add_argument(
        '-m',
        '--mode',
        choices=["Augmentation", "Pipeline"],
        default="Pipeline",
        help='choosing mode to run',
    )
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--stage_2', action='store_true')
    parser.add_argument('--stage_3', action='store_true')
    parser.add_argument('--stage_1', action='store_true')
    parser.add_argument('--aug_type', nargs='+', help='augmentation type used', required=True, default='summary')
    parser.add_argument('--small_test',action='store_true', help='Use small test data for Tabfact dataset')
    parser.add_argument('--verbose', action='store_true', help="whether verbose output from llm chain")
    # parser.add_argument('--dubug', type=bool, help='Debug using sample data', default=True)
    parser.add_argument('--use_sample', action='store_true', help='Use sample data')
    parser.add_argument('--save_file', action='store_true', help='Save output in file')
    parser.add_argument('--model', type=str, default='gpt-3.5-turbo-0125', help='LLM Model')
    # parser.add_argument('--k_shot', type=int, default=1, help='Number of k-shot.')
    # parser.add_argument(
    #     '-c', '--config', default='config.json', help='Path to the config file.'
    # )

    # Parse the arguments
    args = parser.parse_args()

    return args


if __name__ == "__main__":

    args = add_arguments()
    print(args)
    if args.mode == 'Pipeline':
        new_pipeline(
            task_name=args.task_name,
            split=args.split,
            model_name=args.model,
            use_sample=args.use_sample,
            small_test=args.small_test,
            verbose = args.verbose,
            save_file=args.save_file,
            aug_type= args.aug_type,
            batch_size = args.batch_size)
        
    if args.mode == 'Augmentation':
        augmentation(
            task_name=args.task_name,
            split=args.split,
            model_name=args.model,
            use_sample=args.use_sample,
            aug_type=args.aug_type,
            batch_size = args.batch_size,
            small_test=args.small_test
            
        )
