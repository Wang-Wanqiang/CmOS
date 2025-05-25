# Beyond the Textual: Generating Coherent Visual Options for MCQs
Cross-modal Options Synthesis (CmOS) is a multimodal educational question generation with visual options framework, integrating Multimodal Chain-of-Thought (MCoT) reasoning with Retrieval-Augmented Generation (RAG) to generate MCQs with visual options. Specifically, the framework employs an Multimodal Large Language Model (MLLM) to encode multimodal content and embeds it into a four-stage MCoT architecture that separates content discrimination, question and reason generation, alternative pairs screening, and visual options generation. To improve the quality of visual options, we leverage RAG to retrieve similar images from an external educational image database as templates for generation, and then the MLLM and the T2I model are required to optimize based on the templates.

# Requirements
python 3.11 and above

# Quickstart
Below, we provide simple examples to show how to use CmOS.
Before running the code, make sure you have setup the environment and installed the required packages. Make sure you meet the above requirements, and then install the dependent libraries.
```python
pip install -r requirements.txt
```
To determine whether a question is suitable for conversion into a visual multiple-choice format using a multimodal large language model (MLLM), you can run the following script:
```python
python discrimination.py
```
Prerequisites:

1. exemplar.csv — exemplar questions dataset  
2. test.csv — test questions dataset
3. configured MLLM API access (e.g., API key)
