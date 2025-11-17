#!/bin/bash -l
#SBATCH --job-name=pl-ddp-slurm
#SBATCH --account=p200177
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4       # 4 tasks -> 4 Lightning processes per node
#SBATCH --gpus-per-task=1         # 1 GPU per process
#SBATCH --cpus-per-task=8
#SBATCH --time=36:00:00
#SBATCH --qos=default
#SBATCH -p gpu


set -euo pipefail

module load env/release/2024.1
module load git/2.45.1-GCCcore-13.3.0
module load Python/3.12.3-GCCcore-13.3.0 
 
# activate your venv
source /mnt/tier2/project/p200177/u101329/diffusion-interp/.venv/bin/activate



# optional allocator / comms tuning
export OMP_NUM_THREADS=8
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
export NCCL_DEBUG=WARN
# If your fabric is IB, this is often right on MeluXina; if comms hang, try uncommenting the next line
export NCCL_SOCKET_IFNAME=ib0
# export NCCL_IB_DISABLE=1

# ensure the DDP rendezvous port is fixed (Slurm provides MASTER_ADDR/PORT, but a fallback helps)
export MASTER_PORT=${MASTER_PORT:-29501}



echo "Job $SLURM_JOB_ID on $(hostname). Launching $SLURM_NTASKS tasks..."
echo "Node sees GPUs:"
nvidia-smi -L || true

# Launch 4 *separate* processes; Slurm sets CUDA_VISIBLE_DEVICES per task automatically
HYDRA_FULL_ERROR=1 SLURM_NNODES=$SLURM_NNODES srun --gpu-bind=single:1 --cpu-bind=cores  \
  python3 -u train.py -cn ddp_config.yaml
