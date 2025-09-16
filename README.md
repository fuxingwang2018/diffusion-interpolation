# Difuusion Interpolation


## Interactive node
```bash
salloc -A p200177  -p gpu --qos default -N 1 -t 1:00:00
````
## Generate the dataset

```bash
# Sample labels.csv
#Name,Importance,PosX,PosY,Date,LeadTime,Member
# 2023/01/01/00/2023010100_lt00_mem000.npy,1,256,256,2023-01-01T00:00:00Z,0,0
# 2023/01/01/00/2023010100_lt00_mem001.npy,1,256,256,2023-01-01T00:00:00Z,0,1
# 2023/01/01/00/2023010100_lt00_mem002.npy,1,256,256,2023-01-01T00:00:00Z,0,2
# 2023/01/01/00/2023010100_lt00_mem003.npy,1,256,256,2023-01-01T00:00:00Z,0,3
# 2023/01/01/00/2023010100_lt00_mem004.npy,1,256,256,2023-01-01T00:00:00Z,0,4

module load env/release/2024.1
module load Python/3.12.3-GCCcore-13.3.0
python tools/get_sequences_csv.py --labels labels.csv --windows 0-6,6-12,12-18 --members 0,1,2 --start-date 2023-01-01T00:00:00Z --end-date 2023-03-01T00:00:00Z --out  sequences-test.csv --extra-out  sequences-for-stats-test.csv --no-verify-fs

python tools/calc_stats.py --file-list sequences-for-stats-test.csv  --root-dir samples --out sequences-stats-test.npz
```

### Files used for test
All files are in dir `/home/users/u101329/p200177_t2/DE_371/datasets/datasets_SMHI/npy_intep`

```
sequences-for-stats-test.csv
labels.csv
sequences-test.csv
sequences-stats-test.npz
samples/
```


## Build the image
```bash
module load Apptainer/1.3.6-GCCcore-13.3.0
apptainer build --fakeroot ../containers/container.sif  container.def
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


