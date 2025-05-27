import os
import sys

from alpaca_farm.auto_annotations import PairwiseAutoAnnotator
from itertools import combinations
import concurrent.futures
from tqdm import tqdm
import re  # Add import for regular expressions
import random


def split_solution_to_sp_user_prompt(solution_str):
    parts = solution_str.split("<｜Assistant｜>")
    
    # Extract the prompt part (before <|Assistant|>)
    sp_user_prompt = parts[0].strip()
    
    # Extract the response part (after <|Assistant|>)
    cot_response = parts[1].strip()

    only_sp_user_prompt = sp_user_prompt.split("<｜begin▁of▁sentence｜><｜User｜>")[1]
    only_cot_response = cot_response.split("<｜end▁of▁sentence｜>")[0]


    # assert False
    return only_sp_user_prompt, only_cot_response

def split_cot_response(response):
    # Extract everything before </think> as the chain of thought
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

def compute_pair_score(response_pair, rm_version, tokenizer):
    annotator = PairwiseAutoAnnotator()
    
    index_1, index_2 = response_pair['index_1'], response_pair['index_2']
    sp_user_prompt = response_pair['sp_user_prompt']
    response_1, response_2 = response_pair['response_1'], response_pair['response_2']

    cot_1, answer_1 = split_cot_response(response_1)
    cot_2, answer_2 = split_cot_response(response_2)

    cot_1_token = tokenizer.encode(cot_1)
    cot_2_token = tokenizer.encode(cot_2)

    # cot_1_len = cot_1_token.shape[-1]
    # cot_2_len = cot_2_token.shape[-1]
    cot_1_len = len(cot_1_token)
    cot_2_len = len(cot_2_token)

    answer_1_token = tokenizer.encode(answer_1)
    answer_2_token = tokenizer.encode(answer_2)

    # answer_1_len = answer_1_token.shape[-1]
    # answer_2_len = answer_2_token.shape[-1]
    answer_1_len = len(answer_1_token)
    answer_2_len = len(answer_2_token)

    pair_data = {
                'instruction': sp_user_prompt,
                'input': "",  # Empty input field
                'output_1': answer_1,
                'output_2': answer_2
            }

    annotated = annotator.annotate_pairs(to_annotate=[pair_data])
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

    beta_constant = 0
    if rm_version == 'v0':
        beta_constant = 0 
    elif rm_version == 'v1':
        beta_constant = 1
    else:
        raise ValueError(f"Invalid rm_version: {rm_version}")

    beta = beta_constant if len(cot_1) < len(cot_2) else -beta_constant
    if len(cot_1) == len(cot_2):
        beta = 0
    
    # Format penalty: -10 if response doesn't contain "<think>" and "</think>" or cot or answer is too long
    print(f"cot_1_len: {cot_1_len}, answer_1_len: {answer_1_len}")
    print(f"cot_2_len: {cot_2_len}, answer_2_len: {answer_2_len}")
    format_penalty_1 = -100 if ("<think>" not in response_1 or "</think>" not in response_1 or cot_1_len>512 or answer_1_len>512) else 0
    format_penalty_2 = -100 if ("<think>" not in response_2 or "</think>" not in response_2 or cot_2_len>512 or answer_2_len>512) else 0

    score_1, score_2 = 0, 0

    if alpha > 0:
        if beta > 0:
            score_1, score_2 = alpha + beta, -alpha - beta
        elif beta <= 0: 
            score_1, score_2 = alpha, -alpha

    elif alpha == 0:
        score_1, score_2 = beta, -beta

    elif alpha < 0:
        if beta >= 0:
            score_1, score_2 = alpha, -alpha
        elif beta < 0:
            score_1, score_2 = alpha + beta, -alpha - beta
    

    if format_penalty_1 == -100:
        print(f"response_1: {response_1} with format penalty {format_penalty_1}")
        score_1 = format_penalty_1
    if format_penalty_2 == -100:
        print(f"response_2: {response_2} with format penalty {format_penalty_2}")
        score_2 = format_penalty_2

    # score_1 = alpha+beta if format_penalty_1 == 0 else format_penalty_1
    # score_2 = -alpha-beta if format_penalty_2 == 0 else format_penalty_2

    print(f"len(cot_1): {len(cot_1)}, len(cot_2): {len(cot_2)}, score_1: {score_1}, score_2: {score_2}, preference: {preference}")

    return index_1, index_2, score_1, score_2
    # except Exception as e:
    #     print(f"Error in compute_pair_score: {e}")
    #     # Return a default score to avoid breaking the entire process
    #     return response_pair['index_1'], response_pair['index_2'], 0

def compute_score_alpaca(
    solution_str, 
    ground_truth,
    *args,
    **kwargs
):

    rm_version = kwargs['rm_version'] if 'rm_version' in kwargs else 'v0'
    max_response_length = kwargs['max_response_length'] if 'max_response_length' in kwargs else None
    response_length_list = kwargs['response_length_list'] if 'response_length_list' in kwargs else None
    tokenizer = kwargs['tokenizer'] if 'tokenizer' in kwargs else None
    
    solution_group = solution_str
    ground_truth_group = ground_truth
    sp_user_prompt_group = []
    response_group = []

    for solution_str in solution_group:
        sp_user_prompt, response = split_solution_to_sp_user_prompt(solution_str)
        sp_user_prompt_group.append(sp_user_prompt)
        response_group.append(response)

    # Check if all elements in a list are the same
    assert all(x == sp_user_prompt_group[0] for x in sp_user_prompt_group), "Not all sp_user_prompt elements are the same"

    sp_user_prompt = sp_user_prompt_group[0]
    # Create pairwise combinations of responses with their indices
    response_pairs = []

    print(f"response group length:{len(response_group)}")
    print(f'rm version:{rm_version}')
    if rm_version in ['v0', 'v1', 'v2', 'v3']:
        for i, j in combinations(range(len(response_group)), 2):
            response_pairs.append({
                'index_1': i,
                'index_2': j,
                'sp_user_prompt': sp_user_prompt,
                'response_1': response_group[i],
                'response_2': response_group[j]
            })
    elif rm_version in ['v0_2', 'v1_2', 'v2_2', 'v3_2']:
        for i in range(len(response_group)):
            next_i = (i + 1) % len(response_group)  # This makes the last element connect with the first
            response_pairs.append({
                'index_1': i,
                'index_2': next_i,
                'sp_user_prompt': sp_user_prompt,
                'response_1': response_group[i],
                'response_2': response_group[next_i]
            })
    else:
        raise NotImplementedError
    
    print(f"Created {len(response_pairs)} pairwise combinations")
    
    # Process pairs in parallel
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(2, len(response_pairs))) as executor:
        future_to_pair = {executor.submit(compute_pair_score, pair, rm_version, tokenizer): pair for pair in response_pairs}
        
        # for future in tqdm(concurrent.futures.as_completed(future_to_pair), total=len(response_pairs), desc="Computing pairwise scores"):
        for future in concurrent.futures.as_completed(future_to_pair):
            pair = future_to_pair[future]
            # try:
            index_1, index_2, score_1, score_2 = future.result()
            results.append((index_1, index_2, score_1, score_2))
            # except Exception as e:
            #     print(f"Error processing pair {pair['index_1']}-{pair['index_2']}: {e}")
    
    # Process results to create final scores
    scores = [0] * len(response_group)
    for index_1, index_2, score_1, score_2 in results:
        scores[index_1] += score_1
        scores[index_2] += score_2
    
    return scores, ""

    