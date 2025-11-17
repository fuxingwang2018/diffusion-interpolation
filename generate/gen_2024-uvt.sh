#!/bin/bash

set -e 
set -x 


export DATA_DIR=/mnt/tier1/project/p200177/u101329/npy_interp
export OUT_DIR=/mnt/tier1/project/p200177/u101329/results/samples_utv_2024 #

export PYTHONPATH=`pwd`:$PYTHONPATH
range=${1:ERROR}
device=${2:ERROR}


module load env/release/2024.1
module load Python/3.12.3-GCCcore-13.3.0 
source .venv/bin/activate
python generate/sample.py --config-path `pwd`/conf -cn ddp_config_generate.yaml \
    ckpt_file='/mnt/tier2/project/p200177/u101329/diffusion-interp/_work/ddpm-npy-uvt-20250929_203652/epoch-419-val_loss-0.002.ckpt' \
    outdir=${OUT_DIR} \
    datamodule.sequences_csv=${DATA_DIR}/sequences-2023-2024.csv  \
    datamodule.date_range='["2024-10-22","2024-12-31"]' \
    datamodule.members="${members}" \
    device=${device}
