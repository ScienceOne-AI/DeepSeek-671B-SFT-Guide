#!/bin/bash
#SBATCH --job-name=deepseek_vllm              # 任务名称 (Task Name)
#SBATCH --output=./slurm_log/output_%j.log    # 输出文件 (包含任务ID的日志文件) (Output File)
#SBATCH --error=./slurm_log/error_%j.log      # 错误文件 (包含任务ID的日志文件) (Error File)
#SBATCH --nodes=4                             # 节点数 (Node Num)
#SBATCH --gres=gpu:8                          # 每个节点 GPU 数量 (GPU Num)
#SBATCH --ntasks-per-node=1                   # 任务数量 (Task Num)
#SBATCH --time=1-1:00:00                      # 最大运行时间 (格式: 天-小时:分钟:秒) (Max Time of Running)
#SBATCH --mem=500G                            # 分配的内存大小 (Memory Size)
#SBATCH --partition=nvidia-A100               # 指定分区 (Partition)

# load env
source /nfs/miniconda3/etc/profile.d/conda.sh
conda activate vllm

# ip configs
export HEAD_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export DASHBOARD_PORT=8265
export HEAD_PORT=6379

# ray
export RAY_TMPDIR=/tmp/ray_lqx/
export RAY_ADDRESS=$HEAD_ADDR:$HEAD_PORT

RAY_HEAD_BASH="
ray start --block --head --port=$HEAD_PORT --dashboard-port=$DASHBOARD_PORT --temp-dir=$RAY_TMPDIR
"
RAY_WORKER_BASH="
ray start --block --address=$HEAD_ADDR:$HEAD_PORT
"
VLLM_SERVE_BASH="
vllm serve /path/of/your/deepseek_sft_ckpt \
    --tensor-parallel-size 8 \
    --pipeline-parallel-size 4 \
    --served-model-name deepseek-r1-sft \
    --max-model-len 32768 \
    --trust-remote-code \
    --enable-reasoning \
    --reasoning-parser deepseek_r1
"

echo "Starting Ray Head Node on $HEAD_ADDR"
srun --nodes=1 --ntasks=1 --exclusive bash -c "$RAY_HEAD_BASH" &
sleep 10  # wait for the head node to start

echo "Starting Ray Worker Nodes"
srun --nodes=$((SLURM_JOB_NUM_NODES - 1)) --ntasks=$((SLURM_JOB_NUM_NODES - 1)) --exclusive bash -c "$RAY_WORKER_BASH" &
sleep 10  # wait for the worker nodes to start

echo "Ray cluster is ready. Starting vLLM Serve..."
bash -c "$VLLM_SERVE_BASH"

echo "Serving job completed at $(date)"
