# Difuusion Interpolation


## Build the image
```bash
module load Apptainer/1.3.6-GCCcore-13.3.0
apptainer build --fakeroot ../test-diffusion-interp/container.sif  container.def
```

## Start MLFlow

```bash
module load Python/3.12.3-GCCcore-13.3.0
pip3 intall --user mlflow
python3 -m mlflow server \
  --host localhost\
  --port 5000 \
  --backend-store-uri sqlite:///./mlruns/mlflow.db  
   
```

## Run the training
```bash
module load env/release/2024.1
module load PyTorch/2.3.0-foss-2024a-CUDA-12.6.0 
module load Apptainer/1.3.6-GCCcore-13.3.0

DATA_DIR=/home/users/u101329/p200177_t2/DE_371/datasets/datasets_SMHI/npy_intep
DATA_DIR=/home/users/u101329/p200177_t2/DE_371/datasets/datasets_SMHI/npy_intep/minst

SIF_FILE=../test-diffusion-interp/container.sif 
srun --ntasks=4 --gpus-per-task=1  apptainer exec \
    --nv \
    --containall \
    --bind ../test-diffusion-interp:/work \
    --bind /dev/shm:/dev/shm \
    --bind $DATA_DIR/samples:/data \
    --bind .:/code \
    $SIF_FILE \
    bash -c "
        set -e
        cd /code
        nvidia-smi
        python train.py
    "
```


