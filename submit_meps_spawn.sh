#!/bin/bash -l
#SBATCH --job-name=pt-ddp-spawn
#SBATCH --account=p200177
#SBATCH --nodes=1
#SBATCH --ntasks=1                 # single process
#SBATCH --gpus-per-node=4          # all 4 GPUs visible to that process
#SBATCH --cpus-per-task=32         # give enough CPU for dataloader workers across 4 ranks
#SBATCH --time=00:30:00
#SBATCH --qos=short
#SBATCH -p gpu
set -euo pipefail

ln -sf test_meps_ddp_spawn.yaml conf/config.yaml

module load env/release/2024.1
module load Apptainer/1.3.6-GCCcore-13.3.0
module load git/2.45.1-GCCcore-13.3.0

export DATA_DIR=/home/users/u101329/p200177_t2/DE_371/datasets/datasets_SMHI/npy_intep
export SIF_FILE=../containers/container.sif

# optional allocator tuning (MeluXina may ignore expandable_segments)
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export NCCL_DEBUG=WARN
# If IB causes issues on this node, you can try:
# export NCCL_IB_DISABLE=1

apptainer exec --nv --containall \
  --env PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF} \
  --env CUDA_DEVICE_ORDER=${CUDA_DEVICE_ORDER} \
  --env NCCL_DEBUG=${NCCL_DEBUG} \
  --bind ./work:/work \
  --bind "${DATA_DIR}/samples":/samples \
  --bind "${DATA_DIR}":/data \
  --bind .:/code \
  "${SIF_FILE}" \
  bash -lc '
    set -e
    cd /code
    echo "Outside Lightning (parent) sees GPUs:"
    nvidia-smi -L || true
    python -u train.py
  '
