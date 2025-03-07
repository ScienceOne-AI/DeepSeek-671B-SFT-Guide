#!/bin/bash

# get the IP address / hostname of node0 as NODE_0_ADDR
NODE_0_ADDR=$(ssh node12 "hostname -I | cut -d ' ' -f 1")

# set the total number of nodes and the number of GPUs per node
NNODES=32
NPROC_PER_NODE=8
PORT=10088

# set the node rank for each node
HOSTNAME=$(hostname)

# get the node rank, assuming the hostname format is node0, node1, ..., node31
NODE_RANK=$(echo $HOSTNAME | sed 's/[^0-9]*\([0-9]*\)/\1/' | awk '{print $1}')

echo 
echo ===================================
echo master  IP          : $NODE_0_ADDR
echo current node        : $HOSTNAME
echo current NODE_RANK   : $NODE_RANK
echo ===================================
echo

# load miniconda
export PATH="/nfs/miniconda3/bin:$PATH"
source /nfs/miniconda3/etc/profile.d/conda.sh
conda activate ds_env
cd /nfs/projects/yayi_xtuner

# environment variables
export NCCL_IB_HCA=mlx5_0:1,mlx5_3:1,mlx5_4:1,mlx5_5:1,mlx5_6:1,mlx5_8:1,mlx5_9:1,mlx5_10:1
export NCCL_IB_DISABLE=0
export NCCL_SOCKET_IFNAME=ens19f0np0
export NCCL_DEBUG=INFO

export NODE_0_ADDR
export OMP_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
export NPROC_PER_NODE=$NPROC_PER_NODE
export NNODES=$NNODES
export PORT=$PORT
export ADDR=$NODE_0_ADDR
export NODE_RANK=$NODE_RANK

# start training
xtuner train ./code/scripts/sft_deepseek \
    --deepspeed deepspeed_zero3_offload \
    --work-dir "/path/of/your/deepseek_sft_ckpt"