# QueryAgent: A Reliable and Efficient Reasoning Framework with Environmental Feedback based Self-Correction

---
 [![Contributions Welcome](https://img.shields.io/badge/Contributions-Welcome-brightgreen.svg?style=flat-square)](https://github.com/dki-lab/GrailQA/issues)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![language-python3](https://img.shields.io/badge/Language-Python3-blue.svg?style=flat-square)](https://www.python.org/)
[![made-with-Pytorch](https://img.shields.io/badge/Made%20with-Pytorch-orange.svg?style=flat-square)](https://pytorch.org/)
[![paper](https://img.shields.io/badge/Paper-ACL2024-lightgrey?style=flat-square)](https://arxiv.org/abs/2310.15517)
<img width="1175" alt="image" src="https://raw.githubusercontent.com/cdhx/img_store/main/queryagent.png">

> Employing Large Language Models (LLMs)
for semantic parsing has achieved remarkable
success. However, we find existing methods fall short in terms of reliability and efficiency when hallucinations are encountered. In
this paper, we address these challenges with a
framework called QueryAgent, which solves
a question step-by-step and performs stepwise self-correction. We introduce an environmental feedback-based self-correction method
called ERASER. Unlike traditional approaches,
ERASER leverages rich environmental feedback in the intermediate steps to perform selective and differentiated self-correction only
when necessary. Experimental results demonstrate that QueryAgent notably outperforms all
previous few-shot methods using only one example on GrailQA and GraphQ by 7.0 and 15.0
F1. Moreover, our approach exhibits superiority in terms of efficiency, including runtime,
query overhead, and API invocation costs. By
leveraging ERASER, we further improve another baseline (i.e., AgentBench) by approximately 10 points, revealing the strong transferability of our approach



# KB employment
You can follow this to employee Freebase in your local device:  
https://github.com/dki-lab/GrailQA?tab=readme-ov-file#setup

# Run
1. config the hyper-parameters in `agent_utils/config.py`
2. fill the api-key and the KB query endpoint in `agent_utils/config.py`
3. execute `agent_utils/run_exp.py`

**Note:** Due to some policy restrictions, 
we can't open source the openai embedding file, 
you can run one yourself based on the code in `similarity_search.py->get_openai_embedding`, 
The cost of this part is very low, 
less than $1 for all the dataset's relations and questions with the use of cache.
# File Structure

```
QueryAgent/
├── ag_src: source code directory 
|  ├── agent_utils: some tool functions file 
|     ├── ag_utils.py: the tool function of Agent framework
|     ├── similarity_search.py: the relation ranking module
|     ├── run_exp.py: the main function entrance, excute this file to 
|     └── config.py: the configuration file for experiment
|  ├── grail_src: code for GrailQA experiment
|     ├── GRAIL.py: the main function of GrailQA experiment
|     ├── sparql_generator.py: the PyQL compiler(the action set of QueryAgent) for GrailQA
|     └── wikienv.py: the detail function of GrailQA experiment 
|  ├── graphq_src: code for GraphQ experiment
|     └──  the file struct of other dataset is similar to grail_src 
|  ├── webqsp_src: code for WebQSP experiment
|  └── meta_src: code for MetaQ experiment
├── data: the dataset for experiment
|     ├──GrailQA_v1.0    
|     ├──GraphQ
|     ├──metaQA
|     └──WebQSP
```

# Citation

```
@misc{huang2024queryagent,
      title={QueryAgent: A Reliable and Efficient Reasoning Framework with Environmental Feedback based Self-Correction}, 
      author={Xiang Huang and Sitao Cheng and Shanshan Huang and Jiayu Shen and Yong Xu and Chaoyun Zhang and Yuzhong Qu},
      year={2024},
      eprint={2403.11886},
      archivePrefix={arXiv},
      primaryClass={cs.CL}

```

