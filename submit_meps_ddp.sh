#!/bin/bash -l
#SBATCH --job-name=pt-ddp
#SBATCH --account=p200177
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=8 
#SBATCH --mem=0
#SBATCH --time=00:30:00
#SBATCH --qos=short
#SBATCH -p gpu
 

set -euo pipefail

module load env/release/2024.1
module load git/2.45.1-GCCcore-13.3.0
module load Python/3.12.3-GCCcore-13.3.0 
 
# activate your venv
source .venv/bin/activate

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





# point Hydra config (if you swap configs via symlink)
ln -sf test_meps_ddp.yaml conf/config.yaml

echo "Job $SLURM_JOB_ID on $(hostname). Launching $SLURM_NTASKS tasks..."
echo "Node sees GPUs:"
nvidia-smi -L || true

# Launch 4 *separate* processes; Slurm sets CUDA_VISIBLE_DEVICES per task automatically
srun python3 -u train.py
