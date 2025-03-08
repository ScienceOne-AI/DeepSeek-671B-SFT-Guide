# DeepSeek-V3/R1-617B Full Parameter Fine-Tuning Guide

<div align="center" style="line-height: 1;">

[![GitHub Stars](https://img.shields.io/github/stars/ScienceOne-AI/DeepSeek-671B-SFT-Guide?style=social)](https://github.com/ScienceOne-AI/DeepSeek-671B-SFT-Guide)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

[中文版](./README_zh.md) ｜ [English](./README.md)

</div>

An open-source solution for full parameter fine-tuning of DeepSeek-V3/R1 671B, including complete code and scripts from training to inference, as well as some practical experiences and conclusions, jointly launched by the Institute of Automation of the Chinese Academy of Sciences and Beijing Wenge Technology Co. Ltd.

## 🌟 Project Highlights
- Implemented modeling files containing DeepSeek-V3/R1 training logic (see `./model`, code logic completed based on Deepseek-V3 paper and Deepseek-V2 modeling files);
- Implemented full parameter fine-tuning of DeepSeek-V3/R1 671B based on data parallelism (DeepSpeed ZeRO) + sequence parallelism (SP);
- Summarized the entire process of model training and deployment, including pitfalls, encountered problems, and solutions.

## 🚀 Quick Start
### 1. Hardware Configuration

The configuration of a single server is shown in the table below. There are 32 machines with the same configuration in the cluster, sharing 100TB of storage space, mounted at `/nfs`. The operating system of the machines is Ubuntu 22.04, with IB network communication between machines, NVLink communication between GPUs, and CUDA version 12.6.

| Component  | Specification/Version       | Command to View Details  |  
|------------|-----------------------------|--------------------------| 
| GPU        | 8 x NVIDIA H100 80GB HBM3   | `nvidia-smi`             |   
| CPU        | Intel(R) Xeon(R) Platinum 8463B (96 Cores) | `lscpu` |    
| Memory     | 2.0TB DDR4                  | `free -h`                | 
| Storage    | 100TB NVMe SSD              | `df -h`                  | 
| Network    | InfiniBand 400G             | `ibstat`                 | 
| OS         | Ubuntu 22.04                | `uname -a`               | 
| CUDA       | CUDA 12.6                   | `nvcc -V`                | 

### 2. Environment Setup

We extended and improved the xtuner framework to support full parameter fine-tuning of Deepseek V3/R1 (i.e., `DeepseekV3ForCausalLM` model architecture), supporting data parallelism (DeepSpeed ZeRO based DP) and sequence parallelism (SP).

Install the Python environment, install dependencies according to the `requirements.txt` in the project, and overwrite the core code related to `DeepseekV3ForCausalLM` training in `./code/xtuner` to the corresponding code in the original xtuner package.

```bash
conda create -n ds_env python=3.10
conda activate ds_env
pip install -r requirements.txt

# Overwrite core code, modify to your environment path
YOUR_ENV_PATH='/nfs/miniconda3/envs/ds_env/lib/python3.10/site-packages'
cp -r ./code/xtuner $YOUR_ENV_PATH
```

### 3. Data Preparation
We extended the OpenAI standard data format to be compatible with reasoning data. Each original training data is formatted as follows. If there is a reasoning process, the `reasoning_content` field of the assistant role is not empty.
```json
{
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "User question"},
    {"role": "assistant", "content": "Final answer", "reasoning_content": "Reasoning process"}
  ]
}
```

To simplify the processing logic, we merged the `reasoning_content` and `content` into the `content` field according to the Deepseek training format. Additionally, to be compatible with multi-turn dialogue training logic, we added a `loss` field for each round of the assistant role, and only calculate the loss for `content` with `loss` value `true`.
```json
{
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "User question"},
    {"role": "assistant", "content": "<think>\nReasoning process\n</think>\n\nFinal answer", "loss": true}
  ]
}
```

To clearly show the data storage format, we provide a sample file of the converted data `./data/train_example.json` for reference.

During actual training, the program will automatically convert to the following format according to the Deepseek V3/R1 training template, here for display only:
```
<｜begin▁of▁sentence｜>You are a helpful assistant.<｜User｜>User question<｜Assistant｜><think>\nReasoning process\n</think>\n\nFinal answer<｜end▁of▁sentence｜>
```

### 4. Start Training
We provide training code and training startup scripts, including:
- `./code/scripts/sft_deepseek.py`: Configuration file for sft training, including hyperparameter settings, model and tokenizer configuration, training strategy, etc. Modify model training-related configurations in this file.
- `./code/scripts/sft_deepseek.sh`: sft training startup script, which is an execution file for a single node, so it needs to be executed on each machine through slurm or pdsh. For each machine, the only difference in the training startup command is the `NODE_RANK` value. If there are 32 machines, the number ranges from 0 to 31.

Using pdsh as an example, the steps to start training are as follows:
1. Overwrite the `modeling_deepseek.py` file provided in the `./model` directory of this project to the corresponding original file downloaded from platforms like huggingface;
2. Use pdsh to start training, execute the command `pdsh -R ssh -w node[0-31] 'bash ./code/scripts/sft_deepseek.sh'` on the 0th machine to start the full parameter fine-tuning task of the model on 32 machines. Modify `node[0-31]` according to your machine hostname or IP address. During training, you can visualize the training process (loss changes, etc.) through tensorboard.

----
Below are the conclusions of several experiments we conducted, including the feasibility of model training under different parallel strategies and configurations. The training data is ~100k, and the training context length is 32k. The table reports the number of machines used (nodes), sequence parallelism (sp), data parallelism method (dp), single card batch size (bs), number of iterations (epoch), learning rate (lr), single card memory (mem), experiment records, and notes.

| nodes | sp  | dp             | bs |epoch| lr   | mem   | notes       |
|-------|-----|----------------|----|----|-------|-------|-------------|
| 16    | 8   | zero3_offload  | 2  | 1  | 2e-7  | ~30GB  | ✅ Trainable |
| 32    | 8   | zero3_offload  | 1  | 1  | 1e-5  | ~32GB  | ✅ Trainable |
| 32    | 4   | zero3_offload  | 1  | 1  | 2e-7  | ~25GB  | ✅ Trainable |
| 32    | 1   | zero3_offload  | 1  | 1  | 2e-7  | ~30GB  | ✅ Trainable |
| 32    | 4   | zero3_offload  | 2  | 1  | 2e-7  | ~74GB  | ✅ Trainable (Recommended) |
| 32    | 1   | zero3_offload  | 2  | 1  | 2e-7  | OOM    | ❌ Out of Memory |
| 32    | 4   | zero3          | 1  | 1  | 2e-7  | OOM    | ❌ Out of Memory |
| 32    | 1   | zero3          | 1  | 1  | 2e-7  | OOM    | ❌ Out of Memory |

Below is a screenshot during training. We observed that when fully fine-tuning DeepSeek V3 on our constructed reasoning data, the initial loss is usually around 3.5, and after 1 epoch of training, the loss converges to around 1.2.

![Training Log](./log.png)

### 5. Model Weight Conversion

During training, it is recommended to use at least 100TB of SSD large-capacity storage, as a single pth intermediate result occupies about `7.4TB` of disk space. After training, we need to convert the pth to a huggingface format that is better compatible with mainstream inference frameworks (such as vllm). Execute `bash ./code/scripts/convert_pth_to_hf.sh` on a single machine node to complete the model weight format conversion. You can modify the pth path and weight save path in the script according to the actual situation.

Note that this process requires a large amount of CPU memory, so you can expand it through virtual memory to prevent Out-of-memory. Swap (swap partition) is Linux's virtual memory, which is used to store part of the data to the disk when the physical memory (RAM) is insufficient, freeing up RAM.

```bash
sudo fallocate -l 8192G /tmp/swapfile  # Create 8T swap file
sudo chmod 600 /tmp/swapfile
sudo mkswap /tmp/swapfile
sudo swapon /tmp/swapfile
free -h  # Check if swap has increased
```

### 6. Model Inference Deployment
According to the [Deepseek V3 Github](https://github.com/deepseek-ai/DeepSeek-V3?tab=readme-ov-file#6-how-to-run-locally) introduction, there are multiple ways to deploy the model locally. We used [vLLM](https://github.com/vllm-project/vllm) to perform a simple deployment test of the fully fine-tuned model. Here, we assume that an environment named `vllm` has been created according to the vLLM official documentation.

If using a slurm cluster, refer to our provided script and execute the sbatch command `sbatch ./code/scripts/vllm_deploy_slurm.sh` to submit the job. Half-precision (bf16/fp16) models are recommended to be deployed using 4 machines with 32 cards. If you need to configure the port number of ray or api server, you can modify the sh file yourself.

If you need to start the deployment through pdsh (assuming using node0~node3 four machines), refer to the following steps:

1. Set environment variables (node0~node3).
```bash
export HEAD_ADDR="node0"
export DASHBOARD_PORT=8265
export HEAD_PORT=6379
export RAY_TMPDIR=/tmp/ray_tmp/
export RAY_ADDRESS=$HEAD_ADDR:$HEAD_PORT
```

2. Start Ray Head (node0).
```bash
pdsh -R ssh -w node0 "source /nfs/miniconda3/etc/profile.d/conda.sh && conda activate vllm && \
ray start --block --head --port=$HEAD_PORT --dashboard-port=$DASHBOARD_PORT --temp-dir=$RAY_TMPDIR"
```

3. Start Ray Worker (node1~node3).
```bash
pdsh -R ssh -w node1,node2,node3 "source /nfs/miniconda3/etc/profile.d/conda.sh && conda activate vllm && \
ray start --block --address=$HEAD_ADDR:$HEAD_PORT"
```

4. Start vLLM (node0).
```bash
pdsh -R ssh -w node0 "source /nfs/miniconda3/etc/profile.d/conda.sh && conda activate vllm && \
vllm serve /path/of/your/deepseek_sft_ckpt \
    --tensor-parallel-size 8 \
    --pipeline-parallel-size 4 \
    --served-model-name deepseek-r1-sft \
    --max-model-len 32768 \
    --trust-remote-code \
    --enable-reasoning \
    --reasoning-parser deepseek_r1"
```

After starting, you can test whether the interface is started normally through the curl command:

```bash
curl -X POST http://node0:8000/v1/chat/completions -d '{"model": "deepseek-r1-sft", "messages":[{"role":"user", "content": "hello"}]}' -H "Content-Type: application/json"
```

After a while, if the terminal outputs the expected response, it means that the entire process from training to deployment has been successfully completed! 🎉 If there are any problems in the above steps or any suggestions for improvement, please feel free to raise an issue for feedback, and we will try to respond and answer as soon as possible.

## 🤝 Acknowledgements
- DeepSeek-V2/V3/R1: [https://github.com/deepseek-ai](https://github.com/deepseek-ai)
- Huggingface transformers: [https://github.com/huggingface/transformers](https://github.com/huggingface/transformers)
- DeepSpeed: [https://github.com/deepspeedai/DeepSpeed](https://github.com/deepspeedai/DeepSpeed)
- Xtuner: [https://github.com/InternLM/xtuner](https://github.com/InternLM/xtuner)
- vLLM: [https://github.com/vllm-project/vllm](https://github.com/vllm-project/vllm)

## 🔍 License
This project is licensed under the Apache-2.0 License.

## ⭐ Star History
[![Star History Chart](https://api.star-history.com/svg?repos=ScienceOne-AI/DeepSeek-671B-SFT-Guide&type=Date)](https://star-history.com/#ScienceOne-AI/DeepSeek-671B-SFT-Guide&Date)
