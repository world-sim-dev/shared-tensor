#!/bin/bash

export PYTHONPATH=.:$PYTHONPATH
export GPUS_PER_NODE=${GPUS_PER_NODE:-4}

exec torchrun --nproc_per_node=${GPUS_PER_NODE} shared_tensor/server.py
