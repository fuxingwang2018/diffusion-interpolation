#!/bin/bash -l
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:15:00
#SBATCH --partition=gpu
#SBATCH --account=p200177
#SBATCH --qos=short

set -euo pipefail

module load env/release/2024.1
module load Apptainer/1.3.6-GCCcore-13.3.0

DATA_DIR=/home/users/u101329/p200177_t2/DE_371/datasets/datasets_SMHI/npy_intep
SIF_FILE=../containers/container.sif

# One task per GPU; Slurm binds GPUs to tasks
srun --label --export=ALL --gpu-bind=single:1 bash <<'SRUN_SCRIPT'
set -euo pipefail

# Map LOCAL_RANK -> one GPU index from SLURM_STEP_GPUS
if [[ -n "${SLURM_STEP_GPUS:-}" ]]; then
  IFS=',' read -r -a GPUS <<< "${SLURM_STEP_GPUS}"
  CVD="${GPUS[${SLURM_LOCALID:-0}]}"
else
  CVD="${SLURM_LOCALID:-0}"
fi

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="${CVD}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:128

echo "RANK=${SLURM_PROCID} LOCAL_RANK=${SLURM_LOCALID} HOST=${HOSTNAME} STEP_GPUS=${SLURM_STEP_GPUS:-unset} -> CVD=${CUDA_VISIBLE_DEVICES}"

# Launch inside Apptainer; pass env explicitly
apptainer exec \
  --nv \
  --containall \
  --env CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}",CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER}",PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF}" \
  --bind ./work:/work \
  --bind "${DATA_DIR}/samples":/samples \
  --bind "${DATA_DIR}":/data \
  --bind .:/code \
  ../containers/container.sif \
  /bin/bash -s <<'CONTAINER_SCRIPT'
set -euo pipefail
cd /code
echo "Inside container: CVD=${CUDA_VISIBLE_DEVICES}"
nvidia-smi -L

python - <<'PY'
import os, torch
print("[python] RANK=", os.environ.get("SLURM_PROCID"),
      " CVD=", os.environ.get("CUDA_VISIBLE_DEVICES"),
      " cuda_count=", torch.cuda.device_count(),
      " current=", (torch.cuda.current_device() if torch.cuda.device_count() else -1))
PY

python -u train.py
CONTAINER_SCRIPT
SRUN_SCRIPT
