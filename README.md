
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
## 1. Introduction

This is a curated list of papers related to large language models (LLMs), with a particular focus on applications in healthcare, reasoning, and interpretability. For each paper, I provide a brief summary of 2-3 sentences along with my personal interpretation. only show the high quality and latest papers. 

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

### 4.2 Medical Graph RAG: Towards Safe Medical Large Language Model via Graph Retrieval-Augmented Generation
[Paper Link](https://arxiv.org/abs/2408.04187)

## 5. Agent Systems

### 5.1 MDAgents: An Adaptive Collaboration of LLMs for Medical Decision-Making
**NeurIPS 2024 Oral** | [Paper Link](https://arxiv.org/pdf/2404.15155)

This paper:
1. Establishes a dynamic framework that selects either a single agent or a multi-agent collective for multi-round discussions based on the difficulty level of medical queries, ultimately reaching a conclusion. This approach simulates real-world medical decision-making processes.
2. Involves significant engineering effort and costs, as all agents use GPT-4, but the performance is impressive.

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
