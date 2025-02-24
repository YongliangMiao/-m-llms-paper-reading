# health&&(m)llms  
# paper list

This is a list of papers related to **health&&(m)llms**, including some of the latest papers. For each paper, I will write a brief summary of 2-3 sentences, along with my own interpretation. 如果还没看到我的极简的概要感受，那么就是我还在读，先存档后面再写.

## 1. benchmark

Because the medical field is relatively specialized, LLMs may not be familiar with it. Therefore, establishing a benchmark to explore the knowledge boundaries of LLMs is meaningful. Typically, people curate data manually or in a clever way and then conduct tests. At the same time, the testing methods can also be highly innovative.

### 1.1 (2024 NeurIPS Datasets and Benchmarks)&nbsp;&nbsp;*MedCalc-Bench: Evaluating Large Language Models for Medical Calculations*   [link](https://arxiv.org/abs/2406.12036)  
This paper:  1. Defines medical calculation tasks and selects 55 common medical calculation tasks, which is an unexplored area for LLMs in the medical field.&nbsp;&nbsp;&nbsp;&nbsp;2. Manually curates the MEDCALC-BENCH samples.  

### 1.2 (2024 ACL Findings)&nbsp;&nbsp; *Benchmarking Retrieval-Augmented Generation for Medicine* [link](https://arxiv.org/abs/2402.13178)
This paper: 1.Constructs a benchmark for RAG in medical domain question answering, with a comprehensive evaluation approach that combines different retrievers and corpora, primarily focusing on Zero-Shot Learning.
&nbsp;&nbsp;&nbsp;&nbsp;2.Also performs a common test by increasing the number of retrieved snippets and modifying the relative position of the corpus in the prompt to observe performance changes.

## 2. muti-modal
### 2.1 *MEIT: Multi-Modal Electrocardiogram Instruction Tuning on Large Language Models for Report Generation* [link](https://arxiv.org/pdf/2403.04945)


## 3. rag
RAG should be particularly useful in the medical domain, as medical knowledge is highly specialized, and models require more external knowledge. However, there are some issues, such as the quality of the retrieved corpora and how can we ensure that the model is able to understand these corpora? Additionally, the model's preference for different medical corpora may vary.
### 3.1 (2025 Feb) *SearchRAG: Can Search Engines Be Helpful for LLM-based Medical Question Answering?* [link](https://arxiv.org/abs/2502.13233)
this paper: 1. uses real-time engines effectively avoid the outdated drawbacks of static knowledge bases while also mitigates the challenges posed by the complexity of medical information.&nbsp;&nbsp;&nbsp;&nbsp;2.Leverages uncertainty filters out irrelevant information, significantly boosting performance.
### 3.2 *Medical Graph RAG: Towards Safe Medical Large Language Model via Graph Retrieval-Augmented Generation*[link](https://arxiv.org/abs/2408.04187)
## 4. agant
The [EMNLP 2024 tutorial](https://language-agent-tutorial.github.io/) is very helpful to me, and I recommend it as a resource for learning about agents. At the same time, for complex tasks like medical question answering, models are prone to hallucinations, as the data, semantics, and model understanding can often be unclear. In such cases, agents can be quite useful.
### 4.1  (2024 NeurIPS oral) *MDAgents: An Adaptive Collaboration of LLMs for Medical Decision-Making* [link](https://arxiv.org/pdf/2404.15155)
This paper: 1. establishes a dynamic framework that selects either a single agent or a multi-agent collective for multi-round discussions based on the difficulty level of medical queries, ultimately reaching a conclusion. This approach simulates real-world medical decision-making processes. &nbsp;&nbsp;&nbsp;&nbsp;2. involves significant engineering effort and costs, as all agents use GPT-4, but the performance is impressive.

## 5. supervised fine-tuning
Actually, regular direct SFT (Supervised Fine-Tuning) can easily lead to "catastrophic forgetting" and overfitting. The generalization ability of SFT is certainly not as good as some methods that prefer it. SFT is generally considered a way to inject knowledge into the model. If SFT is to be applied, it should either mix specialized data and general knowledge data in a refined ratio, or use other very clever methods.
### 5.1  (2025 Feb) *FineMedLM-o1: Enhancing the Medical Reasoning Ability of LLM from Supervised Fine-Tuning to Test-Time Training* [link](https://arxiv.org/pdf/2501.09213)
### 5.2  (2024 COLM) *RAFT: Adapting Language Model to Domain Specific RAG* [link](https://arxiv.org/abs/2403.101313)
## 6. reinforcement learning,  preference-based optimization
### 6.1 (2025 Feb) *Aligning LLMs to Ask Good Questions A Case Study in Clinical Reasoning* [link](https://arxiv.org/abs/2502.14860)
## 7. uncertainty
### 7.1 (2025 Feb) *Ask Patients with Patience: Enabling LLMs for Human-Centric Medical Dialogue with Grounded Reasoning* [link](https://arxiv.org/pdf/2502.07143)
### 7.2 (2025 Feb) *SearchRAG: Can Search Engines Be Helpful for LLM-based Medical Question Answering?* [link](https://arxiv.org/pdf/2502.13233)
