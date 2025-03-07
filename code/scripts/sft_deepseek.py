"""
Data format:

[{
    "messages": [
        { "role": "system", "content": "xxx." },
        { "role": "user", "content": "xxx." },
        { "role": "assistant", "content": "xxx.", "loss": false},
        { "role": "user", "content": "xxx." },
        { "role": "assistant", "content": "xxx.", "loss": true}
    ]
},
...
]
"""
import torch
from datasets import load_dataset
from mmengine.dataset import DefaultSampler
from mmengine.hooks import (CheckpointHook, DistSamplerSeedHook, IterTimerHook,
                            LoggerHook, ParamSchedulerHook)
from mmengine.optim import AmpOptimWrapper, CosineAnnealingLR, LinearLR
from torch.optim import AdamW
from transformers import AutoTokenizer

from xtuner.dataset import process_hf_dataset
from xtuner.dataset.collate_fns import default_collate_fn
from xtuner.dataset.map_fns import openai_map_fn, template_map_fn_factory
from xtuner.engine.hooks import (DatasetInfoHook, HFCheckpointHook,
                                 ThroughputHook,
                                 VarlenAttnArgsToMessageHubHook)
from xtuner.engine.runner import TrainLoop
from xtuner.model import SupervisedFinetune
from xtuner.model.transformers_models.deepseek_v3 import DeepseekV3ForCausalLM
from xtuner.parallel.sequence import SequenceParallelSampler
from xtuner.utils import PROMPT_TEMPLATE
from mmengine.visualization import Visualizer, TensorboardVisBackend

#######################################################################
#                          PART 1  Settings                           #
#######################################################################
pretrained_model_name_or_path = 'model/DeepSeek-V3-bf16'  # change to your model path

# Data
data_files = [
    'data/train_example.json'
]
prompt_template = PROMPT_TEMPLATE.deepseek_v3
max_length = 32768 
max_position_embeddings = 32768

pack_to_max_length = False  # if True, pack to max_length, else pack to the longest sequence
use_varlen_attn = False     # if True, use varlen_attention

# parallel
sequence_parallel_size = 4

# Scheduler & Optimizer
batch_size = 2              # per_device
accumulative_counts = 4     # bs = 1 GPU * batch_size_per_device * accumulative_counts
accumulative_counts *= sequence_parallel_size

dataloader_num_workers = 0
max_epochs = 1
optim_type = AdamW
lr = 2e-7
betas = (0.9, 0.999)
weight_decay = 0
max_norm = 1  # grad clip
warmup_ratio = 0.03

# Save
save_steps = 100
save_total_limit = 2  # Maximum checkpoints to keep (-1 means unlimited)

#######################################################################
#                      PART 2  Model & Tokenizer                      #
#######################################################################
tokenizer = dict(
    type=AutoTokenizer.from_pretrained,
    pretrained_model_name_or_path=pretrained_model_name_or_path,
    trust_remote_code=True, use_fast=True,
    padding_side='right')

model = dict(
    type=SupervisedFinetune,
    use_varlen_attn=use_varlen_attn,
    max_position_embeddings=max_position_embeddings,
    llm=dict(
        type=DeepseekV3ForCausalLM.from_pretrained,
        pretrained_model_name_or_path=pretrained_model_name_or_path,
        moe_implementation='shard', # if error, try 'origin'
        expert_in_one_shard=8,
        trust_remote_code=True
    ))

#######################################################################
#                      PART 3  Dataset & Dataloader                   #
#######################################################################
sampler = SequenceParallelSampler \
    if sequence_parallel_size > 1 else DefaultSampler

train_dataset = dict(
    type=process_hf_dataset,
    dataset=dict(type=load_dataset, path='json', data_files=data_files, cache_dir="/nfs/data_cache"),
    tokenizer=tokenizer,
    max_length=max_length,
    dataset_map_fn=openai_map_fn,
    template_map_fn=dict(
        type=template_map_fn_factory, template=prompt_template),
    remove_unused_columns=True,
    shuffle_before_pack=True,
    pack_to_max_length=pack_to_max_length,
    use_varlen_attn=use_varlen_attn)

train_dataloader = dict(
    batch_size=batch_size,
    num_workers=dataloader_num_workers,
    dataset=train_dataset,
    sampler=dict(type=sampler, shuffle=True),
    collate_fn=dict(type=default_collate_fn, use_varlen_attn=use_varlen_attn))

#######################################################################
#                    PART 4  Scheduler & Optimizer                    #
#######################################################################
# optimizer
optim_wrapper = dict(
    type=AmpOptimWrapper,
    optimizer=dict(
        type=optim_type, lr=lr, betas=betas, weight_decay=weight_decay),
    clip_grad=dict(max_norm=max_norm, error_if_nonfinite=False),
    accumulative_counts=accumulative_counts,
    loss_scale='dynamic',
    dtype='bfloat16')

# learning policy
param_scheduler = [
    dict(
        type=LinearLR,
        start_factor=1e-5,
        by_epoch=True,
        begin=0,
        end=warmup_ratio * max_epochs,
        convert_to_iter_based=True),
    dict(
        type=CosineAnnealingLR,
        eta_min=0.0,
        by_epoch=True,
        begin=warmup_ratio * max_epochs,
        end=max_epochs,
        convert_to_iter_based=True)
]

# train, val, test setting
train_cfg = dict(type=TrainLoop, max_epochs=max_epochs)

#######################################################################
#                           PART 5  Runtime                           #
#######################################################################
# Log the dialogue periodically during the training process, optional
custom_hooks = [
    dict(type=DatasetInfoHook, tokenizer=tokenizer),
    dict(type=ThroughputHook),
    dict(type=HFCheckpointHook)
]

# varlen_attention 依赖的 Hook
if use_varlen_attn:
    custom_hooks += [dict(type=VarlenAttnArgsToMessageHubHook)]

# configure default hooks
default_hooks = dict(
    # record the time of every iteration.
    timer=dict(type=IterTimerHook),
    # print log every 10 iterations.
    logger=dict(type=LoggerHook, log_metric_by_epoch=False, interval=1),
    # enable the parameter scheduler.
    param_scheduler=dict(type=ParamSchedulerHook),
    # save checkpoint per `save_steps`.
    checkpoint=dict(
        type=CheckpointHook,
        by_epoch=False,
        interval=save_steps,
        max_keep_ckpts=save_total_limit),
    # set sampler seed in distributed evrionment.
    sampler_seed=dict(type=DistSamplerSeedHook),
)

# configure environment
env_cfg = dict(
    # whether to enable cudnn benchmark
    cudnn_benchmark=False,
    # set multi process parameters
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    # set distributed parameters
    dist_cfg=dict(backend='nccl'),
)

# set visualizer
visualizer = dict(type=Visualizer, vis_backends=[dict(type=TensorboardVisBackend)])

# set log level
log_level = 'INFO'

# load from which checkpoint
load_from = None

# whether to resume training from the loaded checkpoint
resume = False

# Defaults to use random seed and disable `deterministic`
randomness = dict(seed=None, deterministic=False)

# set log processor
log_processor = dict(by_epoch=False)