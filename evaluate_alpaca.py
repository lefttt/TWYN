import pandas as pd
import json
import os
import time
from tqdm import tqdm
import os
import sys
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from typing import List, Tuple, Dict
import ray
from ray.util.placement_group import placement_group

# from model_merger import load_fsdp_model

from alpaca_farm.auto_annotations import PairwiseAutoAnnotator
from itertools import combinations
import concurrent.futures
from tqdm import tqdm

import torch
import os
import re
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForTokenClassification, AutoModelForVision2Seq

try:
    # for torch 2.5+
    from torch.distributed.tensor import DTensor
    from torch.distributed.tensor import Shard, Placement
except ImportError:
    from torch.distributed._tensor import DTensor
    from torch.distributed._tensor import Shard, Placement


def split_cot_response(response):
    # Extract everything before </think> as the chain of thought

    response = response.split('<｜Assistant｜>')[1].strip()

    # Replace all occurrences of "├" with "|"
    cot = response.split('</think>')[0].strip()
    
    # Use regex to extract content between <answer> and </answer> tags
    answer_match = re.search(r'<answer>(.*?)</answer>', response, re.DOTALL)
    
    # If <answer> tags are present, use that content as the answer
    if answer_match:
        answer = answer_match.group(1).strip()
    # Otherwise, use everything after </think> as the answer
    else:
        answer = response.split('</think>', 1)[1].strip() if '</think>' in response else response.strip()
        
    return cot, answer


def compute_pair_score(response_pair):
    # try:
    annotator = PairwiseAutoAnnotator(annotators_config="alpaca_eval_gpt4")
    
    sp_user_prompt = response_pair['sp_user_prompt']
    response_1, response_2 = response_pair['response_1'], response_pair['response_2']

    cot_1, answer_1 = split_cot_response(response_1)
    cot_2, answer_2 = split_cot_response(response_2)

    pair_data = {
                'instruction': sp_user_prompt,
                'input': "",  # Empty input field
                'output_1': answer_1,
                'output_2': answer_2
            }

    annotated = annotator.annotate_pairs(to_annotate=[pair_data])
    
    # print(f"annotated !!!!!!!!!!!!: {annotated}")

    preference = annotated[0]['preference']
    if preference == 1.0:
        alpha = 5
    elif preference == 2.0:
        alpha = -5
    elif preference == 1.5:
        alpha = 0
    else:
        # print(f"Error: annotated: {annotated}")
        alpha = 0
        # raise ValueError(f"Invalid preference: {preference}")


    return alpha



def merge_by_placement(tensors: List[torch.Tensor], placement: Placement):
    if placement.is_replicate():
        return tensors[0]
    elif placement.is_partial():
        raise NotImplementedError("Partial placement is not supported yet")
    elif placement.is_shard():
        return torch.cat(tensors, dim=placement.dim).contiguous()
    else:
        raise ValueError(f"Unsupported placement: {placement}")



def convert_fsdp_checkpoints_to_hfmodels(local_dir, hf_model_path, target_dir=None):
    """Convert FSDP checkpoints to HuggingFace models without using command-line arguments."""
    # copy rank zero to find the shape of (dp, fsdp)
    rank = 0
    world_size = 0
    for filename in os.listdir(local_dir):
        match = re.match(r"model_world_size_(\d+)_rank_0\.pt", filename)
        if match:
            world_size = match.group(1)
            break
    assert world_size, "No model file with the proper format"

    state_dict = torch.load(os.path.join(local_dir, f'model_world_size_{world_size}_rank_{rank}.pt'),
                            map_location='cpu')
    pivot_key = sorted(list(state_dict.keys()))[0]
    weight = state_dict[pivot_key]

    if isinstance(weight, DTensor):
        # get sharding info
        device_mesh = weight.device_mesh
        mesh = device_mesh.mesh
        mesh_dim_names = device_mesh.mesh_dim_names
    else:
        # for non-DTensor
        mesh = np.array([int(world_size)], dtype=np.int64)
        mesh_dim_names = ('fsdp',)

    print(f'Got device mesh {mesh}, mesh_dim_names {mesh_dim_names}')

    assert mesh_dim_names in (('fsdp',), ('ddp', 'fsdp')), f'Unsupported mesh_dim_names {mesh_dim_names}'

    if 'tp' in mesh_dim_names:
        # fsdp * tp
        total_shards = mesh.shape[-1] * mesh.shape[-2]
        mesh_shape = (mesh.shape[-2], mesh.shape[-1])
    else:
        # fsdp
        total_shards = mesh.shape[-1]
        mesh_shape = (mesh.shape[-1],)

    print(f'Processing model shards with {total_shards} {mesh_shape} in total')

    model_state_dict_lst = []
    model_state_dict_lst.append(state_dict)
    model_state_dict_lst.extend([""] * (total_shards - 1))

    def process_one_shard(rank):
        model_path = os.path.join(local_dir, f'model_world_size_{world_size}_rank_{rank}.pt')
        state_dict = torch.load(model_path, map_location='cpu', weights_only=False)
        model_state_dict_lst[rank] = state_dict
        return state_dict

    with ThreadPoolExecutor(max_workers=min(32, os.cpu_count())) as executor:
        for rank in range(1, total_shards):
            executor.submit(process_one_shard, rank)
    state_dict = {}
    param_placements = {}
    keys = set(model_state_dict_lst[0].keys())
    for key in keys:
        state_dict[key] = []
        for model_state_dict in model_state_dict_lst:
            try:
                tensor = model_state_dict.pop(key)
            except:
                print("-" * 30)
                print(model_state_dict)
            if isinstance(tensor, DTensor):
                state_dict[key].append(tensor._local_tensor.bfloat16())
                placements = tuple(tensor.placements)
                # replicated placement at dp dimension can be discarded
                if mesh_dim_names[0] == 'dp':
                    placements = placements[1:]
                elif mesh_dim_names[0] == 'ddp':
                    placements = placements[1:]
                if key not in param_placements:
                    param_placements[key] = placements
                else:
                    assert param_placements[key] == placements
            else:
                state_dict[key].append(tensor.bfloat16())

    del model_state_dict_lst

    for key in sorted(state_dict):
        if not isinstance(state_dict[key], list):
            print(f"No need to merge key {key}")
            continue
        if key in param_placements:
            # merge shards
            placements = param_placements[key]
            if len(mesh_shape) == 1:
                # 1-D list, FSDP without TP
                assert len(placements) == 1
                shards = state_dict[key]
                state_dict[key] = merge_by_placement(shards, placements[0])
            else:
                # 2-D list, FSDP + TP
                raise NotImplementedError("FSDP + TP is not supported yet")
        else:
            state_dict[key] = torch.cat(state_dict[key], dim=0)

    print('Writing to local disk')
    if target_dir is None:
        hf_path = os.path.join(local_dir, 'huggingface')
    else:
        hf_path = target_dir
    config = AutoConfig.from_pretrained(hf_model_path)

    if 'ForTokenClassification' in config.architectures[0]:
        auto_model = AutoModelForTokenClassification
    elif 'ForCausalLM' in config.architectures[0]:
        auto_model = AutoModelForCausalLM
    elif 'ForConditionalGeneration' in config.architectures[0]:
        auto_model = AutoModelForVision2Seq
    else:
        raise NotImplementedError(f'Unknown architecture {config["architectures"]}')

    with torch.device('meta'):
        model = auto_model.from_config(config, torch_dtype=torch.bfloat16)
    model.to_empty(device='cpu')

    print(f'Saving model to {hf_path}')
    model.save_pretrained(hf_path, state_dict=state_dict)
    del state_dict
    del model
    
    return hf_path

# Ray model inference actor for distributed processing
@ray.remote(num_gpus=1)
class ModelInferenceActor:
    def __init__(self, model_path):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        # Let Ray handle the GPU assignment - don't explicitly set device_id
        self.device = torch.device("cuda")
        print(f"Actor running on device: {self.device}, CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', 'Not Set')}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, 
            torch_dtype=torch.bfloat16
        ).to(self.device)
        self.model.eval()
        
    def infer_batch(self, batch_prompts, max_length=1000):
        import torch
        import re
        
        # Tokenize prompts
        batch_prompts = [f"<｜User｜>{prompt}<｜Assistant｜><think>" for prompt in batch_prompts]
        batch_inputs = self.tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True).to(self.device)
        
        # Run inference
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=batch_inputs["input_ids"],
                attention_mask=batch_inputs["attention_mask"],
                max_length=max_length,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        
        responses = [self.tokenizer.decode(output) for output in outputs]
        return responses
        

def convert_model(fsdp_dir_1, hf_model_path_1, fsdp_dir_2, hf_model_path_2):
    # Step 1: Convert FSDP models to HuggingFace format if needed
    # Model 1
    convert_fsdp_checkpoints_to_hfmodels(fsdp_dir_1, hf_model_path_1)
    # Model 2
    convert_fsdp_checkpoints_to_hfmodels(fsdp_dir_2, hf_model_path_2)
    
def inference_model(saved_model_path_1, saved_model_path_2, input_file, output_file_1, output_file_2, comparison_output_file, max_length=1024):
    # Step 2: Check available Ray resources
    print("\nAnalyzing Ray cluster resources...")
    
    # Get available resources by node
    node_resources = {}
    for node in ray.nodes():
        node_ip = node["NodeManagerAddress"]
        gpus = node["Resources"].get("GPU", 0)
        node_resources[node_ip] = {"gpus": gpus}
        print(f"Node {node_ip}: {gpus} GPUs")
    
    # Count total available GPUs
    available_gpus = sum(node_info["gpus"] for node_info in node_resources.values())
    
    if available_gpus <= 0:
        print("No GPUs found in the Ray cluster. Please check your configuration.")
        return
    
    print(f"Ray cluster has {len(node_resources)} nodes with {available_gpus} total GPUs")
    
    # Step 3: Create distributed Ray actors for inference - allow Ray to handle placement
    num_actors = min(available_gpus, 8)  # Use up to 8 GPUs or as many as available
    print(f"Creating {num_actors} distributed model inference actors")
    
    # Create actors for both models
    actors_model1 = []
    actors_model2 = []
    
    # Let Ray handle actor placement - we specify num_gpus=1 in the remote decorator
    for i in range(num_actors):
        actor1 = ModelInferenceActor.remote(saved_model_path_1)
        actor2 = ModelInferenceActor.remote(saved_model_path_2)
        actors_model1.append(actor1)
        actors_model2.append(actor2)
    
    # Step 4: Process the input file with both models
    print(f"Processing input file: {input_file}")
    
    # Create output directories if they don't exist
    os.makedirs(os.path.dirname(output_file_1), exist_ok=True)
    os.makedirs(os.path.dirname(output_file_2), exist_ok=True)
    os.makedirs(os.path.dirname(comparison_output_file), exist_ok=True)
    
    # Read input file
    with open(input_file, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]
    
    # data = data[:10]
    # Create batches for distributed processing - create more batches than actors for better load balancing
    num_batches = num_actors * 4
    items_per_batch = max(1, len(data) // num_batches)
    batches = [data[i:i + items_per_batch] for i in range(0, len(data), items_per_batch)]
    
    # Process with Model 1
    print(f"Processing with Model 1...")
    results_model1 = []
    pending_tasks_model1 = []
    
    # Submit inference tasks to Ray actors for Model 1
    print(f"Submitting {len(batches)} batches to {num_actors} actors for Model 1")
    for i, batch in enumerate(batches):
        actor_idx = i % len(actors_model1)
        batch_prompts = [item['prompt'][0]['content'] for item in batch]
        pending_tasks_model1.append((batch, actors_model1[actor_idx].infer_batch.remote(batch_prompts, max_length)))
    
    # Process results as they complete for Model 1
    for batch, task in tqdm(pending_tasks_model1, desc="Processing batches for Model 1"):
        try:
            batch_responses = ray.get(task)
            
            # Add responses to batch items
            for i, response in enumerate(batch_responses):
                if i < len(batch):  # Ensure we don't go out of bounds
                    batch[i]['response'] = response
                    
            results_model1.extend(batch)
        except Exception as e:
            print(f"Error processing batch for Model 1: {e}")
    
    # Save Model 1 results to output file
    with open(output_file_1, 'w', encoding='utf-8') as f:
        for item in results_model1:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"Model 1 inference complete! Results saved to {output_file_1}")
    
    # Process with Model 2
    print(f"Processing with Model 2...")
    results_model2 = []
    pending_tasks_model2 = []
    
    # Submit inference tasks to Ray actors for Model 2
    print(f"Submitting {len(batches)} batches to {num_actors} actors for Model 2")
    for i, batch in enumerate(batches):
        actor_idx = i % len(actors_model2)
        batch_prompts = [item['prompt'][0]['content'] for item in batch]
        pending_tasks_model2.append((batch, actors_model2[actor_idx].infer_batch.remote(batch_prompts, max_length)))
    
    # Process results as they complete for Model 2
    for batch, task in tqdm(pending_tasks_model2, desc="Processing batches for Model 2"):
        try:
            batch_responses = ray.get(task)
            
            # Add responses to batch items
            for i, response in enumerate(batch_responses):
                if i < len(batch):  # Ensure we don't go out of bounds
                    batch[i]['response'] = response
                    
            results_model2.extend(batch)
        except Exception as e:
            print(f"Error processing batch for Model 2: {e}")
    
    # Save Model 2 results to output file
    with open(output_file_2, 'w', encoding='utf-8') as f:
        for item in results_model2:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"Model 2 inference complete! Results saved to {output_file_2}")


def inference_single_model(saved_model_path, input_file, output_file, max_length=1024):
    # Step 2: Check available Ray resources
    print("\nAnalyzing Ray cluster resources...")
    
    # Get available resources by node
    node_resources = {}
    for node in ray.nodes():
        node_ip = node["NodeManagerAddress"]
        gpus = node["Resources"].get("GPU", 0)
        node_resources[node_ip] = {"gpus": gpus}
        print(f"Node {node_ip}: {gpus} GPUs")
    
    # Count total available GPUs
    available_gpus = sum(node_info["gpus"] for node_info in node_resources.values())
    
    if available_gpus <= 0:
        print("No GPUs found in the Ray cluster. Please check your configuration.")
        return
    
    print(f"Ray cluster has {len(node_resources)} nodes with {available_gpus} total GPUs")
    
    # Step 3: Create distributed Ray actors for inference - allow Ray to handle placement
    num_actors = min(available_gpus, 8)  # Use up to 8 GPUs or as many as available
    print(f"Creating {num_actors} distributed model inference actors")
    
    # Create actors for the model
    actors = []
    
    # Let Ray handle actor placement - we specify num_gpus=1 in the remote decorator
    for i in range(num_actors):
        actor = ModelInferenceActor.remote(saved_model_path)
        actors.append(actor)
    
    # Step 4: Process the input file with the model
    print(f"Processing input file: {input_file}")
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Read input file
    with open(input_file, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]
    
    # Create batches for distributed processing - create more batches than actors for better load balancing
    num_batches = num_actors * 4
    items_per_batch = max(1, len(data) // num_batches)
    batches = [data[i:i + items_per_batch] for i in range(0, len(data), items_per_batch)]
    
    # Process with the model
    print(f"Processing with model...")
    results = []
    pending_tasks = []
    
    # Submit inference tasks to Ray actors
    print(f"Submitting {len(batches)} batches to {num_actors} actors")
    for i, batch in enumerate(batches):
        actor_idx = i % len(actors)
        batch_prompts = [item['prompt'][0]['content'] for item in batch]
        pending_tasks.append((batch, actors[actor_idx].infer_batch.remote(batch_prompts, max_length)))
    
    # Process results as they complete
    for batch, task in tqdm(pending_tasks, desc="Processing batches"):
        try:
            batch_responses = ray.get(task)
            
            # Add responses to batch items
            for i, response in enumerate(batch_responses):
                if i < len(batch):  # Ensure we don't go out of bounds
                    batch[i]['response'] = response
                    
            results.extend(batch)
        except Exception as e:
            print(f"Error processing batch: {e}")
    
    # Save results to output file
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in results:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"Model inference complete! Results saved to {output_file}")
    return results


def comparison_results(output_file_1, output_file_2, comparison_output_file, comparison_num=None):
    # Step 5: Compare responses from both models and calculate win rates
    print("Comparing responses from both models...")
    
    # Load responses from saved files instead of using in-memory results
    print(f"Loading Model 1 responses from {output_file_1}")
    model1_responses = {}
    with open(output_file_1, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line)
            prompt = item['prompt'][0]['content']
            model1_responses[prompt] = item['response']
    
    print(f"Loading Model 2 responses from {output_file_2}")
    model2_responses = {}
    with open(output_file_2, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line)
            prompt = item['prompt'][0]['content']
            model2_responses[prompt] = item['response']
    
    # Create pairs for comparison using the loaded responses
    comparison_pairs = []
    for prompt in model1_responses:
        if prompt in model2_responses:  # Only compare prompts that both models have responses for
            response_1 = model1_responses[prompt]
            response_2 = model2_responses[prompt]
            
            comparison_pairs.append({
                'sp_user_prompt': prompt,
                'response_1': response_1,
                'response_2': response_2
            })
    
    print(f"Created {len(comparison_pairs)} pairs for comparison")
    if comparison_num is not None:
        comparison_pairs = comparison_pairs[:comparison_num]
        print(f"Comparing {len(comparison_pairs)} pairs")
    
    # Process pairs in parallel
    comparison_results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(16, len(comparison_pairs))) as executor:
        future_to_pair = {executor.submit(compute_pair_score, pair): pair for pair in comparison_pairs}
        
        for future in tqdm(concurrent.futures.as_completed(future_to_pair), total=len(comparison_pairs), desc="Computing pairwise scores"):
            pair = future_to_pair[future]
            try:
                score = future.result()
                cot_1, answer_1 = split_cot_response(pair['response_1'])
                cot_2, answer_2 = split_cot_response(pair['response_2'])
                comparison_results.append({
                    'prompt': pair['sp_user_prompt'],
                    'model1_response': pair['response_1'],
                    'model2_response': pair['response_2'],
                    'score': score,
                    'winner': 'model1' if score > 0 else 'model2' if score < 0 else 'tie',
                    'cot_1': cot_1,
                    'cot_2': cot_2,
                    'answer_1': answer_1,
                    'answer_2': answer_2
                })
            except Exception as e:
                print(f"Error processing pair: {e}")
    
    # Calculate win rates
    model1_wins = sum(1 for result in comparison_results if result['winner'] == 'model1')
    model2_wins = sum(1 for result in comparison_results if result['winner'] == 'model2')
    ties = sum(1 for result in comparison_results if result['winner'] == 'tie')
    total = len(comparison_results)
    
    win_rate_model1 = model1_wins / total if total > 0 else 0
    win_rate_model2 = model2_wins / total if total > 0 else 0
    tie_rate = ties / total if total > 0 else 0
    
    # Calculate average lengths
    avg_cot_1_length = sum(len(result['cot_1']) for result in comparison_results) / total if total > 0 else 0
    avg_cot_2_length = sum(len(result['cot_2']) for result in comparison_results) / total if total > 0 else 0
    avg_answer_1_length = sum(len(result['answer_1']) for result in comparison_results) / total if total > 0 else 0
    avg_answer_2_length = sum(len(result['answer_2']) for result in comparison_results) / total if total > 0 else 0
    
    print(f"Comparison Results:")
    print(f"Model 1 Win Rate: {win_rate_model1:.2%} ({model1_wins}/{total})")
    print(f"Model 2 Win Rate: {win_rate_model2:.2%} ({model2_wins}/{total})")
    print(f"Tie Rate: {tie_rate:.2%} ({ties}/{total})")
    print(f"Average CoT Length - Model 1: {avg_cot_1_length:.2f} chars")
    print(f"Average CoT Length - Model 2: {avg_cot_2_length:.2f} chars")
    print(f"Average Answer Length - Model 1: {avg_answer_1_length:.2f} chars")
    print(f"Average Answer Length - Model 2: {avg_answer_2_length:.2f} chars")
    
    # Save comparison results to output file
    with open(comparison_output_file, 'w', encoding='utf-8') as f:
        # Write summary statistics
        summary = {
            'model1_win_rate': win_rate_model1,
            'model2_win_rate': win_rate_model2,
            'tie_rate': tie_rate,
            'model1_wins': model1_wins,
            'model2_wins': model2_wins,
            'ties': ties,
            'total': total,
            'avg_cot_1_length': avg_cot_1_length,
            'avg_cot_2_length': avg_cot_2_length,
            'avg_answer_1_length': avg_answer_1_length,
            'avg_answer_2_length': avg_answer_2_length
        }
        f.write(json.dumps(summary, ensure_ascii=False) + '\n')
        
        # Write detailed results
        for result in comparison_results:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')
    
    print(f"Comparison results saved to {comparison_output_file}")
    
    # Update or create the summary CSV file
    summary_csv_path = '*.csv'
    
    # Convert summary to row format for CSV
    csv_row = {
        'comparison_file': os.path.basename(comparison_output_file),
        'model1_win_rate': win_rate_model1,
        'model2_win_rate': win_rate_model2,
        'tie_rate': tie_rate,
        'model1_wins': model1_wins,
        'model2_wins': model2_wins,
        'ties': ties,
        'total_comparisons': total,
        'avg_cot_1_length': avg_cot_1_length,
        'avg_cot_2_length': avg_cot_2_length,
        'avg_answer_1_length': avg_answer_1_length,
        'avg_answer_2_length': avg_answer_2_length
    }
    
    # Check if CSV exists and write header if it doesn't
    file_exists = os.path.isfile(summary_csv_path)
    
    with open(summary_csv_path, 'a', newline='', encoding='utf-8') as csvfile:
        fieldnames = csv_row.keys()
        writer = pd.DataFrame([csv_row]).to_csv(csvfile, header=not file_exists, index=False, mode='a')
    
    print(f"Summary information added to {summary_csv_path}")

def main_multal_comparison(exp1_name, exp2_name, steps, hf_model_path):
    # Initialize Ray
    if not ray.is_initialized():
        print("Starting Ray...")
        ray.init(address="auto", ignore_reinit_error=True)
        print("Ray initialized successfully")

    # exp1_name = '0422_1610_1.5b_v0_2_train_alpaca_grpo_gpu32'
    # exp2_name = '0422_1443_1.5b_v2_2_train_alpaca_grpo_gpu32'
    # steps = [100, 200, 300, 400]  # Evaluate these steps
    max_length = 1024
    comparison_num = None
    # hf_model_path = 'deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B'
    hf_model_path = hf_model_path
    
    # Input file path
    input_file = 'test_alpaca.jsonl'
    
    # Evaluate each step
    for step in steps:
        print(f"\n=== Evaluating step {step} ===\n")
        
        # Model 1 paths
        fsdp_dir_1 = f'projects/verl/checkpoints/alpaca/{exp1_name}/global_step_{step}/actor'
        hf_model_path_1 = hf_model_path
        saved_model_path_1 = f'projects/verl/checkpoints/alpaca/{exp1_name}/global_step_{step}/actor/huggingface'
        
        # Model 2 paths
        fsdp_dir_2 = f'projects/verl/checkpoints/alpaca/{exp2_name}/global_step_{step}/actor'
        hf_model_path_2 = hf_model_path
        saved_model_path_2 = f'projects/verl/checkpoints/alpaca/{exp2_name}/global_step_{step}/actor/huggingface'
        
        # Output paths for this step
        output_file_1 = f'projects/verl/.cache/tatsu-lab___alpaca_farm/test_alpaca_response_model1_{exp1_name}_global_step_{step}.jsonl'
        output_file_2 = f'projects/verl/.cache/tatsu-lab___alpaca_farm/test_alpaca_response_model2_{exp2_name}_global_step_{step}.jsonl'
        comparison_output_file = f'projects/verl/.cache/tatsu-lab___alpaca_farm/test_alpaca_{exp1_name}_step_{step}_vvvsss_{exp2_name}_step_{step}_comparison_results.jsonl'
        
        # Uncomment these lines to enable model conversion and inference
        # print(f"Converting models for step {step}...")
        convert_model(fsdp_dir_1, hf_model_path_1, fsdp_dir_2, hf_model_path_2)
        
        # print(f"Running inference for step {step}...")
        inference_model(saved_model_path_1, saved_model_path_2, input_file, output_file_1, output_file_2, comparison_output_file, max_length=max_length)
        
        print(f"Comparing results for step {step}...")
        comparison_results(output_file_1, output_file_2, comparison_output_file, comparison_num=comparison_num)


def main_sft_comparison(exp1_name, exp2_name, steps, sft_model_name):
    # Initialize Ray
    if not ray.is_initialized():
        print("Starting Ray...")
        ray.init(address="auto", ignore_reinit_error=True)
        print("Ray initialized successfully")

    # Input file path
    max_length = 1024
    input_file = 'projects/verl/.cache/tatsu-lab___alpaca_farm/test_alpaca.jsonl'
    #sft_model_name = "DeepSeek-R1-Distill-Qwen-1.5B"
    sft_dir = f"models_HF/{sft_model_name}"
    output_file_sft = f'projects/verl/.cache/tatsu-lab___alpaca_farm/test_alpaca_response_{sft_model_name}.jsonl'
    inference_single_model(sft_dir, input_file, output_file_sft, max_length=max_length)

    #steps = [100, 200, 300, 400]  # Evaluate these steps
    comparison_num = None
    # hf_model_path = 'deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B'
    
    # Evaluate each step
    for step in steps:
        print(f"\n=== Evaluating step {step} ===\n")
        
        # Output paths for this step
        output_file_1 = f'/test_alpaca_response_model1_{exp1_name}_global_step_{step}.jsonl'
        output_file_2 = f'/test_alpaca_response_model2_{exp2_name}_global_step_{step}.jsonl'
        
        print(f"Comparing results for step {step} and model {exp1_name} with sft model {sft_model_name}")
        comparison_output_file_1 = f'/test_alpaca_{exp1_name}_step_{step}_vvvsss_{sft_model_name}_comparison_results.jsonl'
        comparison_results(output_file_1, output_file_sft, comparison_output_file_1, comparison_num=comparison_num)

        print(f"Comparing results for step {step} and model {exp2_name} with sft model {sft_model_name}")
        comparison_output_file_2 = f'/test_alpaca_{exp2_name}_step_{step}_vvvsss_{sft_model_name}_comparison_results.jsonl'
        comparison_results(output_file_2, output_file_sft, comparison_output_file_2, comparison_num=comparison_num)


def main_single_sft_comparison(exp1_name, steps, sft_model_name):
    # Initialize Ray
    if not ray.is_initialized():
        print("Starting Ray...")
        ray.init(address="auto", ignore_reinit_error=True)
        print("Ray initialized successfully")

    # Input file path
    max_length = 1024
    input_file = '/test_alpaca.jsonl'
    #sft_model_name = "DeepSeek-R1-Distill-Qwen-1.5B"
    sft_dir = f"/{sft_model_name}"
    output_file_sft = f'/test_alpaca_response_{sft_model_name}.jsonl'
    inference_single_model(sft_dir, input_file, output_file_sft, max_length=max_length)

    #steps = [100, 200, 300, 400]  # Evaluate these steps
    comparison_num = None
    # hf_model_path = 'deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B'
    
    # Evaluate each step
    for step in steps:
        print(f"\n=== Evaluating step {step} ===\n")
        # Output paths for this step
        output_file_1 = f'/test_alpaca_response_model1_{exp1_name}_global_step_{step}.jsonl'
        # output_file_2 = f'/test_alpaca_response_model2_{exp2_name}_global_step_{step}.jsonl'
        
        print(f"Comparing results for step {step} and model {exp1_name} with sft model {sft_model_name}")
        comparison_output_file_1 = f'/test_alpaca_{exp1_name}_step_{step}_vvvsss_{sft_model_name}_comparison_results.jsonl'
        comparison_results(output_file_1, output_file_sft, comparison_output_file_1, comparison_num=comparison_num)

if __name__ == "__main__":
    ###### 7b results
    exp1_name = ''
    exp2_name = ''
    #steps = [100, 200, 300]  # Evaluate these steps
    steps = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
    ## convert both models to hf models, and then run inference, and then compare the results
    hf_model_path = 'deepseek-ai/DeepSeek-R1-Distill-Qwen-7B'
    main_multal_comparison(exp1_name, exp2_name, steps, hf_model_path)

    sft_model_name = "DeepSeek-R1-Distill-Qwen-7B"
    ## run inferece for the sft model, and then compare the results with the two models
    # main_sft_comparison(exp1_name, exp2_name, steps, sft_model_name)
    main_single_sft_comparison(exp1_name, steps, sft_model_name)


    ###### 1.5b results
    exp1_name = ''
    exp2_name = ''
    #steps = [100, 200, 300]  # Evaluate these steps
    steps = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
    ## convert both models to hf models, and then run inference, and then compare the results
    hf_model_path = 'deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B'
    main_multal_comparison(exp1_name, exp2_name, steps, hf_model_path)

    sft_model_name = "DeepSeek-R1-Distill-Qwen-1.5B"
    ## run inferece for the sft model, and then compare the results with the two models
    # main_sft_comparison(exp1_name, exp2_name, steps, sft_model_name)
    main_single_sft_comparison(exp1_name, steps, sft_model_name)


