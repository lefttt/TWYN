import re
import os
import datasets
from tqdm import tqdm
import json
import re
import time
import sys
import json
import random
import argparse
from typing import Dict, Any, Optional

SP_MATH_0305 = """Let's think step by step,write the thought in <think> and </think>,then output the final answer within \\boxed{}."""
# add a row to each data item that represents a unique id

def make_map_fn(split: str,dataset_name):
    """Create a mapping function to process dataset examples.

    Args:
        split: Dataset split name ('train' or 'test')

    Returns:
        Function that processes individual dataset examples
    """
    def process_fn(example: Dict[str, Any], idx: int) -> Optional[Dict[str, Any]]:
        question = example.pop('problem')
        instruction = SP_MATH_0305
        question = f"{question} {instruction}"
        answer = example.pop('answer')

        data = {
            "data_source": dataset_name,
            "prompt": [{
                "role": "user",
                "content": question
            }],
            "ability": "math",
            "reward_model": {
                "style": "rule",
                "ground_truth": answer
            },
            "extra_info": {
                'split': split,
                'index': idx
            }
        }
        return data
    return process_fn

def construct_rl_prompt(args):

    train_dataset = datasets.load_dataset('json', data_files=args.local_train_path)
    # 将source为aime和omni的数据剔除掉
    train_dataset = train_dataset.filter(lambda x: x['source'] not in ['amc_aime', 'Omni-MATH'])
    # 将原本只有train的数据划分为train和eval
    train_eval_split = train_dataset['train'].train_test_split(test_size=0.2)  
    train_dataset = train_eval_split['train']
    eval_dataset = train_eval_split['test']
    
    train_dataset = train_dataset.map(function=make_map_fn('train', args.dataset_name), with_indices=True)
    eval_dataset = eval_dataset.map(function=make_map_fn('eval', args.dataset_name), with_indices=True)
    # print("Available train splits:", train_dataset.keys())
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    train_dataset.to_parquet(os.path.join(args.output_dir, f'train_{args.version}_{args.dataset_name}.parquet'))
    eval_dataset.to_parquet(os.path.join(args.output_dir, f'eval_{args.version}_{args.dataset_name}.parquet'))

def make_dapo_map_fn(split: str,dataset_name):
    """Create a mapping function to process dataset examples.

    Args:
        split: Dataset split name ('train' or 'test')

    Returns:
        Function that processes individual dataset examples
    """
    def process_fn(example: Dict[str, Any], idx: int) -> Optional[Dict[str, Any]]:
        question = example.pop('prompt')[0]["content"]
        question = question.split("is the answer to the problem.\n\n",1)[1].rsplit("\n\nRemember to put your answer on its own line",1)[0]
        instruction = SP_MATH_0305
        question = f"{question} {instruction}"
        answer = example.pop('reward_model')["ground_truth"]

        data = {
            "data_source": dataset_name,
            "prompt": [{
                "role": "user",
                "content": question
            }],
            "ability": "math",
            "reward_model": {
                "style": "rule",
                "ground_truth": answer
            },
            "extra_info": {
                'split': split,
                'index': idx
            }
        }
        return data
    return process_fn

def create_math_dapo_dataset(args):
    train_dataset = datasets.load_dataset('parquet', data_files=args.local_train_path)
    train_dataset = train_dataset.map(function=make_dapo_map_fn('train', args.dataset_name), with_indices=True)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    train_dataset["train"].to_parquet(os.path.join(args.output_dir, f'train_{args.version}_{args.dataset_name}.parquet'))




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_train_path', default='')
    parser.add_argument('--output_dir', default='')
    parser.add_argument('--dataset_name', default='')
    parser.add_argument('--version', default='')
    
    args = parser.parse_args()
    # construct_rl_prompt(args)
    create_math_dapo_dataset(args)