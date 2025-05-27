# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Note that we don't combine the main with ray_trainer as ray_trainer is used by other main.
"""
import threading
import concurrent
from concurrent.futures import ThreadPoolExecutor

from tqdm import tqdm

from verl import DataProto
import torch
from verl.utils.reward_score import custom_math, gsm8k, math_sympy, alpaca
from verl.trainer.ppo.ray_trainer import RayPPOTrainer


def _default_compute_score(data_source, solution_str, ground_truth, *args, **kwargs):
    if data_source == 'openai/gsm8k':
        return gsm8k.compute_score(solution_str, ground_truth)
    elif data_source in ['lighteval/MATH', 'DigitalLearningGmbH/MATH-lighteval']:
        return custom_math.compute_score(solution_str, ground_truth)
    elif data_source in ["math","big_math","dapo_math","aime","amc","minerva","olympiad_bench"]:
        return math_sympy.math_sympy_reward_fn(solution_str, ground_truth, *args, **kwargs)
    elif data_source == 'train_alpaca':
        return alpaca.compute_score_alpaca(solution_str, ground_truth, *args, **kwargs)
    else:
        raise NotImplementedError


class RewardManager():
    """The reward manager.
    """

    def __init__(self, tokenizer, num_examine, compute_score=None, gen_rm=False, **kwargs) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.compute_score = compute_score or _default_compute_score
        self.gen_rm = gen_rm
        self.grpo_n = kwargs['grpo_n'] if 'grpo_n' in kwargs else None
        self.rm_version = kwargs['rm_version'] if 'rm_version' in kwargs else None
        self.format_penalty = kwargs['format_penalty'] if 'format_penalty' in kwargs else None
        self.max_response_length = kwargs['max_response_length'] if 'max_response_length' in kwargs else None
        
    def process_alpaca_item(self, data_group, already_print_data_sources, i):
        # Get the batch size from the shape of the prompts
        batch_size = data_group.batch['prompts'].shape[0]
        
        # Extract individual items
        prompt_ids_group = []
        valid_prompt_length_group = []
        valid_response_length_group = []
        sequences_str_group = []
        
        # Get the data source - it's a NumPy array, so we need to access the first element
        data_source = data_group.non_tensor_batch['data_source'][0]
        
        # Get the ground truth - it's also a NumPy array of dictionaries
        ground_truth = data_group.non_tensor_batch['reward_model'][0]['ground_truth']
        
        for j in range(batch_size):
            # Extract prompt IDs for this item
            prompt_ids = data_group.batch['prompts'][j]
            prompt_length = prompt_ids.shape[-1]
            prompt_ids_group.append(prompt_ids)
            
            # Calculate valid lengths
            valid_prompt_length = data_group.batch['attention_mask'][j, :prompt_length].sum().item()
            valid_prompt_length_group.append(valid_prompt_length)
            
            valid_response_length = data_group.batch['attention_mask'][j, prompt_length:].sum().item()
            valid_response_length_group.append(valid_response_length)
            
            # Get valid prompt IDs
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]
            
            # Get response IDs
            response_ids = data_group.batch['responses'][j, :valid_response_length]
            
            # Create sequence
            sequences = torch.cat((valid_prompt_ids, response_ids))
            sequences_str = self.tokenizer.decode(sequences)
            sequences_str_group.append(sequences_str)

        
        # Compute scores for the group
        score_group, rm_response_total = self.compute_score(
            data_source=data_source,
            solution_str=sequences_str_group,
            ground_truth=ground_truth,
            gen_rm=self.gen_rm,
            rm_version=self.rm_version,
            response_length_list=valid_response_length_group,
            max_response_length=self.max_response_length,
            tokenizer=self.tokenizer,
        )
        
        return i, score_group, valid_response_length_group, rm_response_total

    def process_single_item(self, data_item, already_print_data_sources, i):
        prompt_ids = data_item.batch['prompts']
        prompt_length = prompt_ids.shape[-1]
        
        valid_prompt_length = data_item.batch['attention_mask'][:prompt_length].sum()
        valid_prompt_ids = prompt_ids[-valid_prompt_length:]
        
        response_ids = data_item.batch['responses']
        valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()
        valid_response_ids = response_ids[:valid_response_length]
        
        # decode
        sequences = torch.cat((valid_prompt_ids, valid_response_ids))
        sequences_str = self.tokenizer.decode(sequences)
        
        ground_truth = data_item.non_tensor_batch['reward_model']['ground_truth']
        data_source = data_item.non_tensor_batch['data_source']

        # print(sequences_str)
        # print(f"Ground truth:{ground_truth}")
        
        if self.gen_rm:
            score, rm_response_total = self.compute_score(
                data_source=data_source,
                solution_str=sequences_str,
                ground_truth=ground_truth,
            )
        else:
            if self.format_penalty:
                score = self.compute_score(
                    data_source=data_source,
                    solution_str=sequences_str,
                    ground_truth=ground_truth,
                    format_penalty=self.format_penalty,
                )
            else:
                score = self.compute_score(
                    data_source=data_source,
                    solution_str=sequences_str,
                    ground_truth=ground_truth,
                    valid_response_length=valid_response_length,
                )
            print(f"score={score}")
        
        with threading.Lock():
            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0
                
            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                print(sequences_str)
        
        # try:
        #     assert float(score) >= -2
        #     assert float(score) <= 2
        # except:
        #     print("**************************",[score])
        #     raise ValueError


        if self.gen_rm:
            return i, score, valid_response_length, rm_response_total

        else:
            return i, score, valid_response_length


    def math_eval(self, reward_tensor, valid_response_length_list, score_list):
        assert len(valid_response_length_list) == len(score_list)
        for i in range(len(valid_response_length_list)):
            if score_list[i] > 0:
                reward_tensor[i, valid_response_length_list[i] - 1] = 1
            else:
                reward_tensor[i, valid_response_length_list[i] - 1] = 0
        return reward_tensor
    

    def __call__(self, data: DataProto):
        """We will expand this function gradually based on the available datasets"""
        
        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if 'rm_scores' in data.batch.keys():
            return data.batch['rm_scores']

        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)
        already_print_data_sources = {}
        rm_response_list, valid_response_length_list, score_list = {}, {}, {}

        if not self.gen_rm or self.gen_rm == 'offline_compare':
            with ThreadPoolExecutor(max_workers=min(128, len(data))) as executor:
                future_to_item = {
                    executor.submit(
                        self.process_single_item, 
                        data[i], 
                        already_print_data_sources, 
                        i
                    ): i for i in range(len(data))
                }
                
                for future in tqdm(
                    concurrent.futures.as_completed(future_to_item), 
                    total=len(data),
                    desc='Computing reward...'
                ):
                    if self.gen_rm:
                        i, score, valid_response_length, rm_response_total = future.result()
                        rm_response_list[i] = rm_response_total
                    else:
                        i, score, valid_response_length = future.result()
                    valid_response_length_list[i], score_list[i] = valid_response_length, score
                    reward_tensor[i, valid_response_length - 1] = score


        elif self.gen_rm == 'alpaca':
            # we need to group data
            assert len(data) % self.grpo_n == 0
            data_len = len(data)
            group_num = int(data_len / self.grpo_n)

            # print(f"group_num: {group_num}")
            # print(f"data_len: {data_len}")
            # print(data[0:0 + self.grpo_n])

            # Process each group directly without creating full_data_group
            with ThreadPoolExecutor(max_workers=min(64, group_num)) as executor:
                future_to_item = {
                    executor.submit(
                        self.process_alpaca_item, 
                        data[batch_start:batch_start + self.grpo_n],  # Pass the slice directly
                        already_print_data_sources, 
                        i,
                    ): i for i, batch_start in enumerate(range(0, data_len, self.grpo_n))
                }
                
                for future in tqdm(
                    concurrent.futures.as_completed(future_to_item), 
                    total=group_num,
                    desc='Computing groupwise reward...'
                ):
                    i, score_group, valid_response_length_group, rm_response_total = future.result()

                    for j in range(self.grpo_n):
                        reward_tensor[i*self.grpo_n + j, valid_response_length_group[j] - 1] = score_group[j]
                        rm_response_list[i*self.grpo_n + j] = rm_response_total
                        
        else:
            print(f'GenRM {self.gen_rm} not supported!!')

        if self.gen_rm:
            return reward_tensor, rm_response_list
        # if torch.isnan(torch.mean(reward_tensor)).any().item():
        #     print("reward_tensor in RewardManager ===============")
        #     print(list(reward_tensor))
        #     raise ValueError
        return reward_tensor


import ray
import hydra


def get_custom_reward_fn(config):
    import importlib.util, os

    reward_fn_config = config.get("custom_reward_function") or {}
    file_path = reward_fn_config.get("path")
    if not file_path:
        return None

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Reward function file '{file_path}' not found.")

    spec = importlib.util.spec_from_file_location("custom_module", file_path)
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as e:
        raise RuntimeError(f"Error loading module from '{file_path}': {e}")

    function_name = reward_fn_config.get("name")

    if not hasattr(module, function_name):
        raise AttributeError(f"Reward function '{function_name}' not found in '{file_path}'.")

    print(f"using customized reward function '{function_name}' from '{file_path}'")

    return getattr(module, function_name)


@hydra.main(config_path='config', config_name='ppo_trainer', version_base=None)
def main(config):
    run_ppo(config)


def run_ppo(config) -> None:

    if not ray.is_initialized():
        # this is for local ray cluster
        ray.init(runtime_env={'env_vars': {'TOKENIZERS_PARALLELISM': 'true', 'NCCL_DEBUG': 'WARN', 'VLLM_ATTENTION_BACKEND': 'XFORMERS'}})

    ray.get(main_task.remote(config))


@ray.remote(num_cpus=1)  # please make sure main_task is not scheduled on head
def main_task(config):
    from verl.utils.fs import copy_to_local
    # print initial config
    from pprint import pprint
    from omegaconf import OmegaConf
    pprint(OmegaConf.to_container(config, resolve=True))  # resolve=True will eval symbol values
    OmegaConf.resolve(config)

    # download the checkpoint from hdfs
    local_path = copy_to_local(config.actor_rollout_ref.model.path)

    # instantiate tokenizer
    from verl.utils import hf_tokenizer, hf_processor
    tokenizer = hf_tokenizer(local_path)
    processor = hf_processor(local_path, use_fast=True)  # used for multimodal LLM, could be none

    # define worker classes
    if config.actor_rollout_ref.actor.strategy == 'fsdp':
        assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
        from verl.workers.fsdp_workers import ActorRolloutRefWorker, CriticWorker
        from verl.single_controller.ray import RayWorkerGroup
        ray_worker_group_cls = RayWorkerGroup

    elif config.actor_rollout_ref.actor.strategy == 'megatron':
        assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
        from verl.workers.megatron_workers import ActorRolloutRefWorker, CriticWorker
        from verl.single_controller.ray.megatron import NVMegatronRayWorkerGroup
        ray_worker_group_cls = NVMegatronRayWorkerGroup

    else:
        raise NotImplementedError

    from verl.trainer.ppo.ray_trainer import ResourcePoolManager, Role

    role_worker_mapping = {
        Role.ActorRollout: ray.remote(ActorRolloutRefWorker),
        Role.Critic: ray.remote(CriticWorker),
        Role.RefPolicy: ray.remote(ActorRolloutRefWorker)
    }

    global_pool_id = 'global_pool'
    resource_pool_spec = {
        global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
    }
    mapping = {
        Role.ActorRollout: global_pool_id,
        Role.Critic: global_pool_id,
        Role.RefPolicy: global_pool_id,
    }

    # we should adopt a multi-source reward function here
    # - for rule-based rm, we directly call a reward score
    # - for model-based rm, we call a model
    # - for code related prompt, we send to a sandbox if there are test cases
    # - finally, we combine all the rewards together
    # - The reward type depends on the tag of the data
    if config.reward_model.enable:
        if config.reward_model.strategy == 'fsdp':
            from verl.workers.fsdp_workers import RewardModelWorker
        elif config.reward_model.strategy == 'megatron':
            from verl.workers.megatron_workers import RewardModelWorker
        else:
            raise NotImplementedError
        role_worker_mapping[Role.RewardModel] = ray.remote(RewardModelWorker)
        mapping[Role.RewardModel] = global_pool_id

    compute_score = None
    reward_fn = RewardManager(tokenizer=tokenizer, num_examine=0, compute_score=compute_score, 
        gen_rm=config.reward_model.gen_rm, grpo_n=config.actor_rollout_ref.rollout.n, rm_version=config.reward_model.version, 
        format_penalty=config.reward_model.format_penalty, max_response_length=config.data.max_response_length)
    # Note that we always use function-based RM for validation
    val_reward_fn = RewardManager(tokenizer=tokenizer, num_examine=1, compute_score=compute_score, 
        gen_rm=config.reward_model.gen_rm, grpo_n=config.actor_rollout_ref.rollout.n, rm_version=config.reward_model.version, format_penalty=config.reward_model.format_penalty)
    # we do not validate the data.
    # val_reward_fn = None

    resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)

    trainer = RayPPOTrainer(config=config,
                            tokenizer=tokenizer,
                            processor=processor,
                            role_worker_mapping=role_worker_mapping,
                            resource_pool_manager=resource_pool_manager,
                            ray_worker_group_cls=ray_worker_group_cls,
                            reward_fn=reward_fn,
                            val_reward_fn=val_reward_fn)
    trainer.init_workers()
    trainer.fit()


if __name__ == '__main__':
    main()
