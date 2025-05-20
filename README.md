# Think When You Need: Self-Adaptive Chain-of-Thought Learning



[![ArXiv](https://img.shields.io/badge/ArXiv-twyn-yellow?logo=arxiv)](https://arxiv.org/abs/2504.03234)
[![Dataset](https://img.shields.io/badge/huggingface-Dataset-green.svg)](https://huggingface.co/datasets/linke666/twyn/tree/main)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

> This code is based on [VeRL]((https://github.com/volcengine/verl)) framework 

## Key Insights ✨  
- **🧠 Adaptive Reasoning Control** – Dynamically adjusts reasoning depth based on question complexity for optimal efficiency.  
- **⚖️ Dual Reward Optimization** – Balances answer quality and brevity through comparative evaluation.  
- **📊 Versatile Task Support** – Handles both verifiable (fact-based) and fuzzy (open-ended) reasoning tasks.  
- **🔍 Scaling Laws Revealed** – Larger models achieve better results with *shorter* reasoning chains (*"Less is more"* for biger models).  

## Verify Task

![示例图片](img/aime.png)


**Experimental Setup**: 1.5B model with 8K max sequence length

| Benchmark | Response Length (tokens) |  | Accuracy (%) |  |
|------------|-----------|-------------|----------|-------|
|  | **Baseline** | **Ours (Δ%)** | **Baseline** | **Ours** |
| **AIME 2024** | 6,031 | 4,653 🔻22.9 | 28.0 | 28.0 |
| **AMC** | 4,594 | 3,358 🔻26.9 | 65.0 | 63.0 |
| **MATH 500** | 2,567 | 1,480 🔻42.4 | 82.5 | 85.0🟢 |
| **Minerva** | 3,136 | 1,581 🔻49.6 | 26.4 | 27.4🟢 |
| **Olympiad Bench** | 4,360 | 3,323 🔻23.8 | 45.3 | 45.6🟢 |
| **Average** | 4,137 | 2,879 🔻30.4 | 49.4 | 49.8🟢 |


## Fuzzy Task
![示例图片](img/alpacafarm.png)

## Quick Start
### Installation
```bash
cd TWYN
pip install -e .
```

### Datasets


Our raw [Training data](https://huggingface.co/datasets/linke666/twyn/tree/main) is from [DeepScaleR](https://huggingface.co/datasets/agentica-org/DeepScaleR-Preview-Dataset)
and [DAPO](https://huggingface.co/datasets/BytedTsinghua-SIA/DAPO-Math-17k).

[Validation](https://huggingface.co/datasets/linke666/twyn/tree/main): [AIME 2024](https://github.com/lefttt/TWYN/data/evals/0409_aime.parquet),[AMC](https://github.com/lefttt/TWYN/data/evals/0409_amc.parquet),[MATH 500](https://github.com/lefttt/TWYN/data/evals/0409_math.parquet),[Minerva](https://github.com/lefttt/TWYN/data/evals/0409_minerva.parquet),[Olympiad Bench](https://github.com/lefttt/TWYN/data/evals/0409_olympiad_bench.parquet).


### Training Scripts
#### verifiable task
```bash
bash scripts/math/twyn_1.5b_16k_train_dapo_grpo.sh
```
#### fuzzy task
```bash
bash scripts/alpaca/twyn_7b_train_alpaca_grpo_gpu32.sh
```

## Citation
If you use this work in your research, please cite:
```bibtex

```

## License
This project incorporates code from:
- [VERL] (licensed under Apache-2.0)
- [DAPO] (licensed under Apache-2.0)
- [DeepScaleR](licensed under MIT)

The combined work is licensed under [Apache-2.0].

