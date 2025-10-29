#!/bin/bash

set -e 
set -x 


export DATA_DIR=/home/users/u101329/p200177_t1_hp/u101329/npy_interp
export OUT_DIR=/home/users/u101329/p200177_t1_hp/u101329/results/samples_2025_from_rutv_100steps #

export PYTHONPATH=`pwd`:$PYTHONPATH
members=${1:ERROR}
device=${2:ERROR}


module load env/release/2024.1
module load Python/3.12.3-GCCcore-13.3.0 
source .venv/bin/activate
python generate/sample.py --config-path `pwd`/conf -cn ddp_config_generate-4ch.yaml \
    ckpt_file='/home/users/u101329/p200177_t2/u101329/diffusion-interp/_work/ddpm-npy-ruvt-20250929_134920/checkpoints/last.ckpt' \
    outdir="${OUT_DIR}" \
    datamodule.sequences_csv="${DATA_DIR}/sequences-2025.csv" \
    datamodule.date_range='["2025-01-01","2025-01-5"]' \
    datamodule.members="${members}" \
    device="${device}" \
    sampler.steps=100