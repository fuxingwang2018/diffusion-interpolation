#!/bin/bash -l
#SBATCH --nodes=1                          # number of nodes
#SBATCH --ntasks=4                         # number of tasks
#SBATCH --ntasks-per-node=4                # number of tasks per node
#SBATCH --gpus-per-task=1                  # number of gpu per task
#SBATCH --cpus-per-task=1                  # number of cores per task
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
srun apptainer exec \
    --nv \
    --containall \
    --bind ./work:/work \
    --bind $DATA_DIR/samples:/samples \
    --bind $DATA_DIR:/data \
    --bind .:/code \
    $SIF_FILE \
    bash -c "
        set -e
        cd /code       
        python test_meps_numpy_datamodule.py 
    "