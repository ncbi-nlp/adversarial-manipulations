# Adversarial Attacks on Large Language Models in Medicine

## Abstract

The integration of Large Language Models (LLMs) into healthcare applications offers promising advancements in medical diagnostics, treatment recommendations, and patient care. However, the susceptibility of LLMs to adversarial attacks poses a significant threat, potentially leading to harmful outcomes in delicate medical contexts. This study investigates the vulnerability of LLMs to two types of adversarial attacks–prompt injections with malicious instructions and fine-tuning with poisoned samples–across three medical tasks: disease prevention, diagnosis, and treatment. Utilizing real-world patient data, we demonstrate that both open-source and proprietary LLMs are vulnerable to malicious manipulation across multiple tasks. We discover that while integrating poisoned data does not markedly degrade overall model performance on medical benchmarks, it can lead to noticeable shifts in fine-tuned model weights, suggesting a potential pathway for detecting and countering model attacks. This research highlights the urgent need for robust security measures and the development of defensive mechanisms to safeguard LLMs in medical applications, to ensure their safe and effective deployment in healthcare settings. 

## Repository Instructions

### Prerequisites
Below is the list of python packages necessary to reproduce this work, also included in requirement.txt.
- torch 2.7.0
- transformers 4.52.4
- accelerate 1.8.1
- trl 0.19.0
- bitsandbytes 0.46.1
- datasets 3.6.0
- openai 1.85.0

Other packages that are necessary (such as `seaborn`) to run the jupyter notebook is listed in requirement.txt.

### Instructions
All fine-tuning codes are in `script`. For acquiring responses from Azure APIs, performing analysis of the generated data, and creating figures from the statistics, please check different sections of the jupyter notebook in `script`. 

You can call train_lora.py to train model with poison / clean data, with the following parameters:

`python train_lora.py -b {begining x*10% poison data} -e {ending x*10% poison data} -m {model path}`

For example, the code below trains Llama 3.3 70B with 10% poison data to 40% poison data (total of 4 models):

`python train_lora.py -b 1 -e 4 -m meta-llama/Meta-Llama-3.1-70B-Instruct`


You can call `get_finetune.py` to collect the responses of finetuned model, with the following parameters:

` python get_finetune.py -b {begining x*10% poison data} -e {ending x*10% poison data} -m {model path} -s {source result file} -d {target result file} -t {task} -ba {batch size}`

For example, the code below runs the Llama 3.3 70B model trained with 0% to 70% poison data on MIMIC notes using batch size 10, and update it in result.json:

`python get_finetune.py -b 0 -e 7 -m meta-llama/Llama-3.3-70B-Instruct -s result.json -d result.json -t mimic -ba 10`

To get the prompt attack and normal baseline, change the begining to -1.

`python get_finetune.py -b -1 -e -1 -m meta-llama/Llama-3.3-70B-Instruct -s result.json -d result.json -t mimic -ba 10`


