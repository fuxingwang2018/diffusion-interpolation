#!/bin/bash -l
#SBATCH --nodes=1                          # number of nodes
#SBATCH --ntasks=4                         # number of tasks
#SBATCH --ntasks-per-node=4                # number of tasks per node
#SBATCH --gpus-per-task=1                  # number of gpu per task
#SBATCH --cpus-per-task=4                  # number of cores per task
#SBATCH --time=00:15:00                    # time (HH:MM:SS)
#SBATCH --partition=gpu                    # partition
#SBATCH --account=p200177                  # project account
#SBATCH --qos=short                          # SLURM qos ( Now Development)

 
set -e


module load env/release/2024.1
module load Apptainer/1.3.6-GCCcore-13.3.0

set -x 

DATA_DIR=/home/users/u101329/p200177_t2/DE_371/datasets/datasets_SMHI/npy_intep
SIF_FILE=../containers/container.sif 
srun --gpu-bind=single:1  --label bash -lc '
  echo "RANK=$SLURM_PROCID LOCAL_RANK=$SLURM_LOCALID HOST=$HOSTNAME CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
  apptainer exec \
    --nv \
    --containall \
    --env CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES \
    --bind ./work:/work \
    --bind '"$DATA_DIR"'/samples:/samples \
    --bind '"$DATA_DIR"':/data \
    --bind .:/code \
    '"$SIF_FILE"' \
    bash -lc "
      set -e
      cd /code
      nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv -i \$CUDA_VISIBLE_DEVICES
      python -u train.py
    "
'  # End of srun