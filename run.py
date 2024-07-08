import os
import json
import argparse
import numpy as np
from batch_pipe import pipeline
from augmentation import augmentation
import logging


def add_arguments():
    # Create an ArgumentParser object
    parser = argparse.ArgumentParser(
        description='Description of your code running.')

    # Add arguments
    parser.add_argument(
        '-t',
        '--task_name',
        default='tabfact',
        choices=['tabfact', 'wikitable'],
        type=str,
        help='Task name.',
    )
    parser.add_argument(
        '-s',
        '--split',
        default="test",
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
    parser.add_argument(
        '--cache_dir', help='cache directory for saving data', default=None)
    parser.add_argument('--batch_size', type=int, default=16,
                        help='batch size for llm chain')
    parser.add_argument('--aug_type', nargs='+',
                        help='augmentation type used in augmentation process', default='summary')
    parser.add_argument('--small_test', action='store_true',
                        help='Use small test data for Tabfact dataset')
    parser.add_argument('--verbose', action='store_true',
                        help="whether verbose output from llm chain")
    parser.add_argument('--use_sample', action='store_true',
                        help='Use sample data')
    parser.add_argument('--save_file', action='store_true',
                        help='Save output in file')
    parser.add_argument('--model', type=str,
                        default='gpt-3.5-turbo', help='LLM Model')
    args = parser.parse_args()

    return args


if __name__ == "__main__":

    args = add_arguments()
    if args.mode == 'Pipeline':
        pipeline(
            task_name=args.task_name,
            split=args.split,
            model_name=args.model,
            cache_dir=args.cache_dir,
            use_sample=args.use_sample,
            small_test=args.small_test,
            verbose=args.verbose,
            save_file=args.save_file,
            aug_type=args.aug_type,
            batch_size=args.batch_size,
            )

    if args.mode == 'Augmentation':
        augmentation(
            task_name=args.task_name,
            split=args.split,
            model_name=args.model,
            use_sample=args.use_sample,
            aug_type=args.aug_type,
            batch_size=args.batch_size,
            small_test=args.small_test

        )
