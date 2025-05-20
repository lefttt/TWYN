"""
This module contains the RewardMathFn class, which evaluates mathematical answers
and assigns rewards based on their correctness. It utilizes a language model to 
validate answers when necessary.
"""
from typing import List, Union

import os
import sys
try:
    from verl.utils.reward_score.math_utils import extract_answer, grade_answer_sympy, grade_answer_mathd
except:
    from math_utils import extract_answer, grade_answer_sympy, grade_answer_mathd
# from math_utils import extract_answer, grade_answer_sympy, grade_answer_mathd
# from deepscaler.system_prompts import ORM_PROMPT
# from deepscaler.utils import call_gemini_llm, call_oai_rm_llm
ORM_USER_TEMPLATE = """
Problem: {problem}
Answer 1: {answer_1}
Answer 2: {answer_2}
"""

THOUGHT_DELIMITER_START = "<think>"
THOUGHT_DELIMITER_END = "</think>"
# ASSISTANT_DELIMITER = "<｜Assistant｜>"

ASSISTANT_DELIMITER = "<|im_start|>assistant"

def math_sympy_reward_fn(solution_str: str, ground_truth: Union[str, List[str]], *args, **kwargs):
    enable_llm = kwargs.get("enable_llm", False)
    format_penalty = kwargs.get("format_penalty", -1)

    # print(f"solution_str={solution_str}")
    # print("--------------------------------")
    if ASSISTANT_DELIMITER not in solution_str:
        return -1
    model_response = solution_str.split(ASSISTANT_DELIMITER)[1]

    if THOUGHT_DELIMITER_START in model_response and THOUGHT_DELIMITER_END in model_response:
        model_solution = model_response.split(THOUGHT_DELIMITER_END)[1]
    else:
        print(f"think delimiter not found in model_response")
        return format_penalty

    model_answer = extract_answer(model_solution)
    if model_answer is None:
        return format_penalty

    # Process the ground truth(s)
    if ground_truth is None:
        return -1
    
    # Convert single answer to list for uniform processing
    if isinstance(ground_truth, (str, float, int)):
        ground_truths = [ground_truth]
    else:
        ground_truths = ground_truth
        
    # Process each ground truth
    processed_ground_truths = []
    for truth in ground_truths:
        truth = str(truth)
        if "\\boxed" in truth:
            processed_truth = extract_answer(truth)
            if processed_truth is not None:
                processed_ground_truths.append(processed_truth)
        else:
            processed_ground_truths.append(truth)
    
    if not processed_ground_truths:
        return -1

    # Check against all possible correct answers
    for ground_truth in processed_ground_truths:
        is_correct = grade_answer_mathd(model_answer, ground_truth) or grade_answer_sympy(model_answer, ground_truth)
        if is_correct:
            return 1

    # If latex heuristics fail and ORM is enabled, use LLM as ORM to evaluate correctness
    # if enable_llm:
    #     for ground_truth in processed_ground_truths:
    #         try:
    #             orm_response = call_gemini_llm(
    #                 system_prompt=ORM_PROMPT,
    #                 prompt=ORM_USER_TEMPLATE.format(problem=problem, answer_1=model_answer, answer_2=ground_truth),
    #                 temperature=0.0,
    #             )

    #             if "[[YES]]" in orm_response:
    #                 return RewardOutput(reward=self.config.correct_reward, is_correct=True)
    #         except Exception as e:
    #             print ("Error calling Gemini ORM, trying OAI RM")
    #             orm_response = call_oai_rm_llm(
    #                 system_prompt=ORM_PROMPT,
    #                 prompt=ORM_USER_TEMPLATE.format(problem=problem, answer_1=model_answer, answer_2=ground_truth),
    #                 temperature=0.0,
    #                 model_id=OAI_RM_MODEL,
    #             )
                
    #             if "[[YES]]" in orm_response:
    #                 return RewardOutput(reward=self.config.correct_reward, is_correct=True)
    #             continue
            
    return -1



if __name__ == "__main__":
    
    solution_str="Let $P(x)=x^{4}+2 x^{3}-13 x^{2}-14 x+24$ be a polynomial with roots $r_{1}, r_{2}, r_{3}, r_{4}$. Let $Q$ be the quartic polynomial with roots $r_{1}^{2}, r_{2}^{2}, r_{3}^{2}, r_{4}^{2}$, such that the coefficient of the $x^{4}$ term of $Q$ is 1. Simplify the quotient $Q\\left(x^{2}\\right) / P(x)$, leaving your answer in terms of $x$. (You may assume that $x$ is not equal to any of $\\left.r_{1}, r_{2}, r_{3}, r_{4}\\right)$.<think> I am omniscient. </think> The answer is \\boxed{24 + 14*x + (-13)*x^2 - 2*x^3 + x^4}."
    model_response=""
    ground_truth=["10", "$x^{4}-2 x^{3}-13 x^{2}+14 x+24$"]
    solution_str = """<|im_start|>assistant <think> To compare the sizes (or magnitudes) of \(-1\) and \(-2\), we need to understand that the negative sign indicates the numbers are on the negative side of the number line. The further a negative number is from zero, the smaller its magnitude. 

1. The number \(-1\) is 1 unit away from zero.
2. The number \(-2\) is 2 units away from zero.

Since \(-2\) is further from zero than \(-1\), \(-2\) has a larger magnitude but since we are considering the values themselves, \(-2\) is actually smaller than \(-1\). </think>

\\boxed{-2 < -1}<|im_end|>"""
    ground_truth = "-1 > -2"
    output = math_sympy_reward_fn(solution_str, ground_truth, enable_llm=False)
    print(output)
