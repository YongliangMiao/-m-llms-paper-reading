# health&&(m)llms  
# paper list

This is a list of papers related to **health&&(m)llms**, including some of the latest papers. For each paper, I will write a brief summary of 2-3 sentences, along with my own interpretation.

## 1. benchmark

Because the medical field is relatively specialized, LLMs may not be familiar with it. Therefore, establishing a benchmark to explore the knowledge boundaries of LLMs is meaningful. Typically, people curate data manually or in a clever way and then conduct tests. At the same time, the testing methods can also be highly innovative.

### 1.1 (2024 NeurIPS Datasets and Benchmarks)&nbsp;&nbsp;*MedCalc-Bench: Evaluating Large Language Models for Medical Calculations*   [link](https://arxiv.org/abs/2406.12036)  
This paper:  1. Defines medical calculation tasks and selects 55 common medical calculation tasks, which is an unexplored area for LLMs in the medical field.&nbsp;&nbsp;&nbsp;&nbsp;2. Manually curates the MEDCALC-BENCH samples.  

### 1.2 (2024 ACL Findings)&nbsp;&nbsp; *Benchmarking Retrieval-Augmented Generation for Medicine* [link](https://arxiv.org/abs/2402.13178)
This paper: 1.Constructs a benchmark for RAG in medical domain question answering, with a comprehensive evaluation approach that combines different retrievers and corpora, primarily focusing on Zero-Shot Learning.
&nbsp;&nbsp;&nbsp;&nbsp;2.Also performs a common test by increasing the number of retrieved snippets and modifying the relative position of the corpus in the prompt to observe performance changes.

## 2. muti-modal

这一部分列出了一些在深度强化学习领域具有里程碑意义的论文。

### 2.1 *Playing Atari with Deep Reinforcement Learning* by Volodymyr Mnih et al.  
[点击阅读论文](https://arxiv.org/abs/1312.5602)  
代码： [GitHub链接](https://github.com/openai/gym)  
**解读**: 本文提出了通过深度强化学习解决Atari游戏中的策略问题，展示了AI在游戏中达到人类级别表现的可能性。对LLMs在多任务学习中的应用也具有启发性。

### 2.2 *Human-level control through deep reinforcement learning* by Mnih et al.  
[点击阅读论文](https://www.nature.com/articles/nature14236)  
代码： [GitHub链接](https://github.com/DeepMind/pyns)  
**解读**: 这篇论文进一步深入探讨了深度强化学习如何实现人类水平的控制能力，并通过Q-learning算法提供了更为高效的策略优化方法。这为未来在更复杂任务中的应用提供了指导。
