#!/bin/bash
NODES="node[0-31]"

# run on all nodes 
pdsh -R ssh -w $NODES 'bash ./code/scripts/sft_deepseek.py'
