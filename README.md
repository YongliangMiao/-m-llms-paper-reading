
# (m)LLMs Paper List

## Table of Contents
- [1. Introduction](#1-introduction)
- [2. Benchmarks](#2-benchmarks)
  - [2.1 MedCalc-Bench](#21-medcalc-bench-evaluating-large-language-models-for-medical-calculations)
  - [2.2 Benchmarking RAG for Medicine](#22-benchmarking-retrieval-augmented-generation-for-medicine)
- [3. Multi-modal](#3-multi-modal)
  - [3.1 MEIT](#31-meit-multi-modal-electrocardiogram-instruction-tuning)
- [4. Retrieval-Augmented Generation (RAG)](#4-retrieval-augmented-generation-rag)
  - [4.1 SearchRAG](#41-searchrag-can-search-engines-be-helpful-for-llm-based-medical-question-answering)
  - [4.2 Medical Graph RAG](#42-medical-graph-rag-towards-safe-medical-large-language-model-via-graph-retrieval-augmented-generation)
- [5. Agent Systems](#5-agent-systems)
  - [5.1 MDAgents](#51-mdagents-an-adaptive-collaboration-of-llms-for-medical-decision-making)
- [6. Supervised Fine-tuning](#6-supervised-fine-tuning)
  - [6.1 FineMedLM-o1](#61-finemedlm-o1-enhancing-the-medical-reasoning-ability-of-llm)
  - [6.2 RAFT](#62-raft-adapting-language-model-to-domain-specific-rag)
- [7. Alignment & Preference-based Optimization](#7-alignment--preference-based-optimization)
  - [7.1 Aligning LLMs to Ask Good Questions](#71-aligning-llms-to-ask-good-questions-a-case-study-in-clinical-reasoning)
  - [7.2 DeepSeek-R1](#72-deepseek-r1-incentivizing-reasoning-capability-in-llms-via-reinforcement-learning)
- [8. Uncertainty](#8-uncertainty)
  - [8.1 Ask Patients with Patience](#81-ask-patients-with-patience-enabling-llms-for-human-centric-medical-dialogue)
  - [8.2 SearchRAG and Uncertainty](#82-searchrag-can-search-engines-be-helpful-for-llm-based-medical-question-answering)
  - [8.3 To Believe or Not to Believe Your LLM](#83-to-believe-or-not-to-believe-your-llm)
  - [8.4 Understanding Uncertainty via Reasoning Topology](#84-understanding-the-uncertainty-of-llm-explanations-a-perspective-based-on-reasoning-topology)
- [9. LLMs as Judges](#9-llms-as-judges)
- [10. Time-scaling & Long Reasoning](#10-time-scaling--long-reasoning)
  - [10.1 Efficient Test-Time Scaling](#101-efficient-test-time-scaling-via-self-calibration)
  - [10.2 Entropy-based Exploration](#102-entropy-based-exploration-conduction-for-multi-step-reasoning)
  - [10.3 Survey of Long Chain-of-Thought](#103-towards-reasoning-era-a-survey-of-long-chain-of-thought-for-reasoning)
  - [10.4 L1: Controlling Reasoning Time](#104-l1-controlling-how-long-a-reasoning-model-thinks-with-reinforcement-learning)
  - [10.5 Understanding R1-Zero-Like Training](#105-understanding-r1-zero-like-training-a-critical-perspective)

## 1. Introduction

This is a curated list of papers related to large language models (LLMs), with a particular focus on applications in healthcare, reasoning, and interpretability. For each paper, I provide a brief summary of 2-3 sentences along with my personal interpretation. only show the high quality and latest papers. 

## 2. Benchmarks

Because the medical field is relatively specialized, LLMs may not be familiar with it. Therefore, establishing benchmarks to explore the knowledge boundaries of LLMs is meaningful. Typically, researchers curate data manually or employ clever methods and then conduct testing. The testing methodologies themselves can also be highly innovative.

### 2.1 MedCalc-Bench: Evaluating Large Language Models for Medical Calculations
**NeurIPS 2024 Datasets and Benchmarks** | [Paper Link](https://arxiv.org/abs/2406.12036)

This paper:
1. Defines medical calculation tasks and selects 55 common medical calculation tasks, which is an unexplored area for LLMs in the medical field.
2. Manually curates the MEDCALC-BENCH samples for comprehensive evaluation.

### 2.2 Benchmarking Retrieval-Augmented Generation for Medicine
**ACL 2024 Findings** | [Paper Link](https://arxiv.org/abs/2402.13178)

This paper:
1. Constructs a benchmark for RAG in medical domain question answering, with a comprehensive evaluation approach that combines different retrievers and corpora, primarily focusing on Zero-Shot Learning.
2. Also performs a common test by increasing the number of retrieved snippets and modifying the relative position of the corpus in the prompt to observe performance changes.

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

The [EMNLP 2024 tutorial](https://language-agent-tutorial.github.io/) is very helpful for learning about agents. For complex tasks like medical question answering, models are prone to hallucinations, as the data, semantics, and model understanding can often be unclear. In such cases, agents can be quite useful.

### 5.1 MDAgents: An Adaptive Collaboration of LLMs for Medical Decision-Making
**NeurIPS 2024 Oral** | [Paper Link](https://arxiv.org/pdf/2404.15155)

This paper:
1. Establishes a dynamic framework that selects either a single agent or a multi-agent collective for multi-round discussions based on the difficulty level of medical queries, ultimately reaching a conclusion. This approach simulates real-world medical decision-making processes.
2. Involves significant engineering effort and costs, as all agents use GPT-4, but the performance is impressive.

## 6. Supervised Fine-tuning

### 6.1 FineMedLM-o1: Enhancing the Medical Reasoning Ability of LLM
**February 2025** | [Paper Link](https://arxiv.org/pdf/2501.09213)

### 6.2 RAFT: Adapting Language Model to Domain Specific RAG
**COLM 2024** | [Paper Link](https://arxiv.org/abs/2403.101313)

## 7. Alignment & Preference-based Optimization

### 7.1 Aligning LLMs to Ask Good Questions: A Case Study in Clinical Reasoning
**February 2025** | [Paper Link](https://arxiv.org/abs/2502.14860)

### 7.2 DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning
**DeepSeek** | [Paper Link](https://arxiv.org/abs/2501.12948)

Game-changing approach to reinforcement learning for reasoning.

## 8. Uncertainty

### 8.1 Ask Patients with Patience: Enabling LLMs for Human-Centric Medical Dialogue
**February 2025** | [Paper Link](https://arxiv.org/pdf/2502.07143)

### 8.2 SearchRAG: Can Search Engines Be Helpful for LLM-based Medical Question Answering?
**February 2025** | [Paper Link](https://arxiv.org/pdf/2502.13233)

Same as in section 4.1.

### 8.3 To Believe or Not to Believe Your LLM
**DeepMind 2024** | [Paper Link](https://arxiv.org/pdf/2406.02543)

A solid method with extensive mathematical analysis; the first to simultaneously consider both epistemic and aleatoric uncertainties. Derives an information-theoretic metric that enables reliable detection of cases where only epistemic uncertainty is high, indicating that the model's output is unreliable.

### 8.4 Understanding the Uncertainty of LLM Explanations: A Perspective Based on Reasoning Topology
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

### 10.4 L1: Controlling How Long A Reasoning Model Thinks With Reinforcement Learning
**March 2025** | [Paper Link](https://www.arxiv.org/pdf/2503.04697)

Somewhat trivial approach.

### 10.5 Understanding R1-Zero-Like Training: A Critical Perspective
**March 2025** | [Paper Link](https://arxiv.org/abs/2503.20783)

Simple but valid change to the GPTO formula.
