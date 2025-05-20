from datasets import load_dataset, concatenate_datasets
import os
import json

def make_map_fn(split, dataset_name):

    def process_fn(example, idx):

        instruction = example.pop('instruction')
        input = example.pop('input')
        output = example.pop('output')
        # SP_ALPACA_PROMPT = """Let's think step by step,write the thought in <think> and </think>,then output the final answer within <answer> and </answer>."""
        SP_ALPACA_PROMPT = """Let's think step by step,write the thought in <think> and </think>,then output the final answer"""

        data = {
            "data_source": dataset_name,
            "prompt": [
            # {
            #     "role": "system",
            #     "content": "please output your thinking in the following format: <think>...</think>"
            # },
            {
                "role": "user",
                "content": SP_ALPACA_PROMPT + "\n" + instruction + "\n" + input
            }],
            "ability": "chat",
            "reward_model": {
                "style": "rule",
                "ground_truth": output
            },
            "extra_info": {
                'split': split,
                'index': idx
            }
        }
        # print(data)
        return data

    return process_fn

def save_dataset_in_multiple_formats(dataset, output_path):
    """Save dataset in both parquet and jsonl formats with the same base filename."""
    # Save as parquet
    parquet_path = f"{output_path}.parquet"
    dataset.to_parquet(parquet_path)
    print(f"Saved dataset to {parquet_path}")
    
    # Save as jsonl
    jsonl_path = f"{output_path}.jsonl"
    with open(jsonl_path, 'w', encoding='utf-8') as f:
        for item in dataset:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    print(f"Saved dataset to {jsonl_path}")

def construct_rl_prompt():
    output_dir = ""

    # Load datasets
    alpha_ds_eval = load_dataset("tatsu-lab/alpaca_farm", "alpaca_farm_evaluation")
    alpha_ds_train = load_dataset("tatsu-lab/alpaca_farm", "alpaca_instructions")

    # Print available splits to debug
    print("Train splits:", list(alpha_ds_train.keys()))
    print("Eval splits:", list(alpha_ds_eval.keys()))

    # Assign to train and test datasets
    train_dataset = alpha_ds_train
    test_dataset = alpha_ds_eval

    # Process all splits
    train_dataset = train_dataset.map(function=make_map_fn('train', 'train_alpaca'), with_indices=True)
    test_dataset = test_dataset.map(function=make_map_fn('test', 'test_alpaca'), with_indices=True)

    # Combine all training splits into one dataset
    combined_train = train_dataset['sft']
    for split in ['preference', 'unlabeled', 'val']:
        combined_train = concatenate_datasets([combined_train, train_dataset[split]])

    print(f"\nTotal combined size: {len(combined_train)}")
    print(f"Total test size: {len(test_dataset['eval'])}")
    
    # Save combined training data and test data in both formats
    save_dataset_in_multiple_formats(combined_train, os.path.join(output_dir, 'train_alpaca_combined'))
    save_dataset_in_multiple_formats(test_dataset['eval'], os.path.join(output_dir, 'test_alpaca'))

if __name__ == '__main__':
    construct_rl_prompt()
