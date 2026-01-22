# Difuusion Interpolation

## Environment on MiluXina
```bash
# start an interactive session
salloc -A p200177  -p gpu --qos default -N 1 -t 1:00:00

#Create the environment
module load env/release/2024.1
module load Python/3.12.3-GCCcore-13.3.0 
python -m venv .venv 
source .venv/bin/activate
pip3 install --upgrade pip
pip3 install -r requirements.txt

# prepare mlflow directory 
# --backend-store-uri: The same value set to _mlflow_dir in the config files 
python3 -m mlflow server \
  --host localhost\
  --port 5000 \
  --backend-store-uri file:path/to/the/directory

# submit the job 
sbatch submit_meps_ddp.sh
```

## Prepare input data from MEPS npy 
```bash

python make_sequences.py \
  --labels labels.csv \
  --root /mnt/tier2/project/p200177/DE_371/datasets/datasets_SMHI/npy_intep/samples \
  --windows 0-6 \
  --members 0,1 \
  --start-date 2023-01-01T00:00:00Z \
  --end-date 2023-02-28T00:00:00Z \
  --merge-root merged_samples \
  --out sequences-reduced-test.csv  \
  --extra-out sequences-for-stats-reduced-test.csv \
  --no-verify-fs

python correct_merged.py  sequences-reduced-test.csv  --root  merged_samples --out  sequences-reduced-test-corrected.csv 

python make_stats.py --file-list sequences-for-stats-reduced-test.csv --root  merged_samples --out sequences-stats-reduced-test.npz


```
## Build apptainer the image
```bash
module load Apptainer/1.3.6-GCCcore-13.3.0
apptainer build --fakeroot ../containers/container.sif  container.def
```

## Test Data module
 
```bash

 

module load env/release/2024.1
module load Apptainer/1.3.6-GCCcore-13.3.0
module load git
source /mnt/tier2/project/p200177/u101329/diffusion-interp/.venv/bin/activate


ROOT_DIR=/mnt/tier2/project/p200177/u101329/diffusion-interp-phase2
DATA_BIND=/project/home/p200177/u101329/DE371_bis/MEPS_subdomain/
WORK_DIR=$ROOT_DIR/_work
MLFLOW_DIR=/mnt/tier2/project/p200177/u101329/_mlruns_apptainer
SIF_FILE=../containers/container.sif 
apptainer exec \
    --containall \
    --bind .:/code \
    --bind $DATA_BIND:$DATA_BIND \
    --bind $ROOT_DIR:$ROOT_DIR \
    --bind $WORK_DIR:$WORK_DIR \
    --bind $MLFLOW_DIR:$MLFLOW_DIR \
    $SIF_FILE \
    bash -c "
        set -e
        cd /code
        export ROOT_DIR=$ROOT_DIR
        export WORK_DIR=$WORK_DIR
        export MLFLOW_DIR=$MLFLOW_DIR
        export HYDRA_FULL_ERROR=1
        export PYTHONPATH=".:$PYTHONPATH" 
        python test_meps_zarr_datamodule.py -cn  ddp_zarr_config.yaml 
    "
```


## Train on interactive
 
```bash

 

module load env/release/2024.1
module load Apptainer/1.3.6-GCCcore-13.3.0
module load git
source /mnt/tier2/project/p200177/u101329/diffusion-interp/.venv/bin/activate


ROOT_DIR=/mnt/tier2/project/p200177/u101329/diffusion-interp-phase2
DATA_BIND=/project/home/p200177/u101329/DE371_bis/MEPS_subdomain/
WORK_DIR=$ROOT_DIR/_work
MLFLOW_DIR=/mnt/tier2/project/p200177/u101329/_mlruns_apptainer
SIF_FILE=../containers/container.sif 
apptainer exec \
    --nv \
    --containall \
    --bind .:/code \
    --bind $DATA_BIND:$DATA_BIND \
    --bind $ROOT_DIR:$ROOT_DIR \
    --bind $WORK_DIR:$WORK_DIR \
    --bind $MLFLOW_DIR:$MLFLOW_DIR \
    $SIF_FILE \
    bash -c "
        set -e
        cd /code
        export ROOT_DIR=$ROOT_DIR
        export WORK_DIR=$WORK_DIR
        export MLFLOW_DIR=$MLFLOW_DIR
        export HYDRA_FULL_ERROR=1
        export PYTHONPATH=".:$PYTHONPATH" 
        python3 -u train.py -cn ddp_zarr_config.yaml
    "
```


## Sample 

- read the example  [here](generate/README.md)
