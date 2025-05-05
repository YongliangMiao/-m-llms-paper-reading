
# (m)LLMs Paper List

## Table of Contents
- [1. Introduction](#1-introduction)
- [2. Benchmarks](#2-benchmarks)
- [3. Multi-modal](#3-multi-modal)
- [4. Retrieval-Augmented Generation (RAG)](#4-retrieval-augmented-generation-rag)
- [5. Agent Systems](#5-agent-systems)
- [6. Supervised Fine-tuning](#6-supervised-fine-tuning)
- [7. Alignment & Preference-based Optimization](#7-alignment--preference-based-optimization)
- [8. Uncertainty](#8-uncertainty)
- [9. LLMs as Judges](#9-llms-as-judges)
- [10. Time-scaling & Long Reasoning](#10-time-scaling--long-reasoning)
- [11. Latent space reasoning](#11-latent-space-reasoning)
## 1. Introduction

This is a curated list of papers related to large language models (LLMs), with a particular focus on applications in reasoning and interpretability. For each paper, I provide a brief summary of 2-3 sentences along with my personal interpretation. only show the high quality and latest papers. 

## 2. Benchmarks


## 3. Multi-modal

### 3.1 MEIT: Multi-Modal Electrocardiogram Instruction Tuning
[Paper Link](https://arxiv.org/pdf/2403.04945)

## 4. Retrieval-Augmented Generation (RAG)

RAG should be particularly useful in the medical domain, as medical knowledge is highly specialized, and models require more external knowledge. However, there are some issues, such as the quality of the retrieved corpora and how to ensure that the model understands these corpora. Additionally, the model's preference for different medical corpora may vary.

### 4.1 SearchRAG: Can Search Engines Be Helpful for LLM-based Medical Question Answering?
**February 2025** | [Paper Link](https://arxiv.org/abs/2502.13233)

This paper:
1. Uses real-time search engines to effectively avoid the outdated drawbacks of static knowledge bases while also mitigating the challenges posed by the complexity of medical information.
2. Leverages uncertainty to filter out irrelevant information, significantly boosting performance.

### 4.2 Search-R1: Training LLMs to Reason and Leverage Search Engines with Reinforcement Learning 
**Apr 2025** | [Paper Link](https://arxiv.org/pdf/2503.09516)

This shows that RL+rule-based output reward+mask loss (masking out tool results with no gradient backprop) is a viable approach.

## 5. Agent Systems

### 5.1 MDAgents: An Adaptive Collaboration of LLMs for Medical Decision-Making
**NeurIPS 2024 Oral** | [Paper Link](https://arxiv.org/pdf/2404.15155)

This paper:
1. Establishes a dynamic framework that selects either a single agent or a multi-agent collective for multi-round discussions based on the difficulty level of medical queries, ultimately reaching a conclusion. This approach simulates real-world medical decision-making processes.
2. Involves significant engineering effort and costs, as all agents use GPT-4, but the performance is impressive.

### 5.2 Synthetic Data Generation & Multi-Step RL for Reasoning & Tool Use
**2025 Apr** | [Paper Link](https://arxiv.org/pdf/2504.04736)
One of the first batch work(maybe?) to use rl to process learning.
## 6. Supervised Fine-tuning

### 6.1 RAFT: Adapting Language Model to Domain Specific RAG
**COLM 2024** | [Paper Link](https://arxiv.org/abs/2403.101313)


## 7. Alignment & Preference-based Optimization

you can see that in section 10. Time-scaling & Long Reasoning.

## 8. Uncertainty
### 8.1 SearchRAG: Can Search Engines Be Helpful for LLM-based Medical Question Answering?
**February 2025** | [Paper Link](https://arxiv.org/pdf/2502.13233)

Same as in section 4.1.

### 8.2 To Believe or Not to Believe Your LLM
**DeepMind 2024** | [Paper Link](https://arxiv.org/pdf/2406.02543)

A solid method with extensive mathematical analysis; the first to simultaneously consider both epistemic and aleatoric uncertainties. Derives an information-theoretic metric that enables reliable detection of cases where only epistemic uncertainty is high, indicating that the model's output is unreliable.

### 8.3 Understanding the Uncertainty of LLM Explanations: A Perspective Based on Reasoning Topology
**February 2025** | [Paper Link](https://arxiv.org/abs/2502.17026)

A novel way (topology) to quantify uncertainty. Focuses on reasoning uncertainty, an area previous works have not explored yet. However, the precision of this method may be limited.

### 8.4 Reasoning Models Don’t Always Say What They Think
**Apr 3, 2025, Alignment Science Team, Anthropic** | [paper link](https://assets.anthropic.com/m/71876fabef0f0ed4/original/reasoning_models_paper.pdf)

This paper unveils the unfaithfulness of LLMs’ chain-of-thought (CoT) reasoning, a finding that is particularly significant given the current prevalence of rule-based RL methods. One sentence from the paper: “We do not require the model to verbalize literally everything, but a faithful CoT should highlight the key decision factors and reasoning chains that would enable us to predict how it would process similar inputs during deployment. For example, if a sycophantic model bases its prediction on a user’s suggestion, it should verbalize sycophancy.”

## 9. LLMs as Judges

For reference, see this repository: [Awesome-LLMs-as-Judges](https://github.com/CSHaitao/Awesome-LLMs-as-Judges)

## 10. Time-scaling & Long Reasoning

### 10.1 Efficient Test-Time Scaling via Self-Calibration
**March 2025** | [Paper Link](https://arxiv.org/pdf/2503.00031)

This paper focuses on the self-calibration issue and touches on time-scaling. It designs a (query, response, accurate confidence) dataset without human annotations but just by sampling, and trains three small models on this dataset.

### 10.2 Entropy-based Exploration Conduction for Multi-step Reasoning
**March 2025** | [Paper Link](https://arxiv.org/pdf/2503.15848)

A very interesting paper that uses entropy and variance entropy to decide reasoning steps. Impressive approach.

### 10.3 Towards Reasoning Era: A Survey of Long Chain-of-Thought for Reasoning
**March 2025** | [Paper Link](https://arxiv.org/pdf/2503.09567)

### 10.4 DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning
**DeepSeek** | [Paper Link](https://arxiv.org/abs/2501.12948)

Game-changing approach to reinforcement learning for reasoning.
### 10.5 Understanding R1-Zero-Like Training: A Critical Perspective
**March 2025** | [Paper Link](https://arxiv.org/abs/2503.20783)

Simple but valid change to the GPTO formula.

### 10.6 DAPO: An Open-Source LLM Reinforcement Learning System at Scale
**seed March 2025**  | [Paper Link](https://arxiv.org/pdf/2503.14476)
see analysis at [link](https://zhuanlan.zhihu.com/p/31157035727).

### 10.7 AgentRM: Enhancing Agent Generalization with Reward Modeling
**March 2025**  | [Paper Link](https://arxiv.org/pdf/2502.18407)

It's less about a fresh idea and more solid engineering: they used MCTS to grind out a reward model that scores super precisely, setting it up for RL training on the policy.

### 10.8 UNLOCKING EFFICIENT LONG-TO-SHORT LLM REASONING WITH MODEL MERGING
**May 2025**  | [Paper Link](https://arxiv.org/pdf/2503.20641)
Model merging methods applied to 1.5B-scale models, such as TA, Ties-Merging and Sens-Merging, remain effective on simple tasks. Smaller models struggle to learn long CoT reasoning ability through model merging. ////The merging of large-scale models poses significant challenges in simultaneously maintaining reasoning performance while substantially reducing response length. The substantial performance gaps between the merging models likely contribute to this difficulty

## 11. Latent space reasoning
### 11.1 Training Large Language Models to Reason in a Continuous Latent Space
**Yuandong Tian Nov 2024** | [Paper Link](https://arxiv.org/abs/2412.06769)

solved the problem of training in latent space reasoning. But not in a solid way:
1. It's like someone starting a conversation in the middle of their sentence. If you're going to talk like that, at least think through the first part before you speak. The training method in this paper causes the missing reasoning steps in the CoT answer to lose important info.
2.Low Data quality：only 6 steps cot data, bad comparison with 10-100 steps training data 

### 11.2 CODI: Compressing Chain-of-Thought into Continuous Space via Self-Distillation
**Mar 2025** | [Paper Link](https://arxiv.org/pdf/2502.21074)

solved the issue of thorough way to train latent space, but the prompt for training is fixed.

### 12 data, solution space
### 12.1 Does Reinforcement Learning Really Incentivize Reasoning Capacity in LLMs Beyond the Base Model?
[Paper Link](https://arxiv.org/pdf/2504.13837) 
RL 可以高效筛选去学到的推理路径，总体来说还是不错的方案，但可能缺乏探索性。

蒸馏大模型知识可能学的最快，也能学到新东西，但需要依赖更大的模型教学。

### 12.2 LIMO: Less is More for Reasoning
**SJTU, SII, GAIR 2025 Feb 2**  | [Paper Link](https://arxiv.org/pdf/2502.03387)

1.高质量的数据哪怕sft也能放大原有模型的能力，不仅仅是rl. 这点可能还是很震撼的，打破现在大多研究观点，sft只能记忆；sft会破坏推理的观点   2. Less is more，数据distillation等等    类似的还有: [s1](https://arxiv.org/pdf/2501.19393)
