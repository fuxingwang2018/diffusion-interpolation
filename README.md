# Difuusion Interpolation

## Experiments
- See the file `docs/experiments/README.md`
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

## Actual test
python tools/get_sequences_csv.py --labels labels.csv --windows 0-6,6-12,12-18 --start-date 2023-01-01T00:00:00Z --end-date 2023-12-31T00:00:00Z --out  sequences-test2.csv --extra-out  sequences-for-stats-test2.csv --no-verify-fs
python tools/calc_stats.py --file-list sequences-for-stats-test2.csv  --root-dir samples --out sequences-stats-test2.npz

## Actual test
python tools/get_sequences_csv.py --labels labels.csv --windows 0-6,6-12,12-18 --start-date 2023-01-01T00:00:00Z --end-date 2025-12-31T00:00:00Z --out  sequences-test3.csv --extra-out  sequences-for-stats-test3.csv --no-verify-fs
python tools/calc_stats.py --file-list sequences-for-stats-test3.csv  --root-dir samples --out sequences-stats-test3.npz



# 
python tools/get_sequences_csv.py --labels labels.csv --windows 0-6,6-12,12-18,18-24,24-30,30-36 --start-date 2023-01-01T00:00:00Z --end-date 2024-03-31T00:00:00Z --out  sequences-test3.csv --extra-out  sequences-for-stats-test3.csv --no-verify-fs
python tools/calc_stats.py --file-list sequences-for-stats-test3.csv  --root-dir samples --out sequences-stats-test3.npz



python make_sequences.py \
  --labels labels.csv \
  --root /home/users/u101329/p200177_t2/DE_371/datasets/datasets_SMHI/npy_intep/samples \
  --windows 0-6,6-12,12-18,18-24,24-30,30-36,36-42 \
  --start-date 2023-01-01T00:00:00Z \
  --end-date 2024-12-31T00:00:00Z \
  --merge-root merged_samples \
  --out sequences-test5.csv  \
  --extra-out sequences-for-stats-test5.csv \
  --no-verify-fs

python make_stats.py --file-list sequences-for-stats-test5.csv --root-dir merged_samples --out sequences-stats-test5.npz

python correct_merged.py --file-list sequences-test5.csv --root  merged_samples --out  sequences-test5-corrected.csv





python make_sequences.py \
  --labels labels.csv \
  --root /home/users/u101329/p200177_t2/DE_371/datasets/datasets_SMHI/npy_intep/samples \
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
#pip3 intall --user mlflow
python3 -m mlflow server \
  --host localhost\
  --port 5000 \
  --backend-store-uri file:/home/users/u101329/p200177_t2/u101329/_mlruns  
```
- to fowrard the port

```bash
NODE=mel0293 # the interactive node
ssh  $NODE -L 5000:localhost:5000
```


## To not use the container
- Create the environment
```bash
module load env/release/2024.1
module load Python/3.12.3-GCCcore-13.3.0 
python -m venv .venv 
source .venv/bin/activate
pip3 install --upgrade pip
pip3 install -r requirements.txt
``` 

- To run the training
it will automatically link the config file (check the script)
```bash
sbatch submit_meps_ddp.sh
```

## Development training run
 
```bash

salloc -A p200177  -p gpu   --qos default -N 1 -t 5:00:00


module load env/release/2024.1
module load Apptainer/1.3.6-GCCcore-13.3.0
module load git


DATA_DIR=/home/users/u101329/p200177_t1_hp/u101329/npy_interp
ROOT_DIR=/home/users/u101329/p200177_t2/u101329/diffusion-interp
WORK_DIR=$ROOT_DIR/_work
MLFLOW_DIR=/home/users/u101329/p200177_t2/u101329/_mlruns_apptainer
SIF_FILE=../containers/container.sif 
apptainer exec \
    --nv \
    --containall \
    --bind .:/code \
    --bind $DATA_DIR:$DATA_DIR \
    --bind $ROOT_DIR:$ROOT_DIR \
    --bind $WORK_DIR:$WORK_DIR \
    --bind $MLFLOW_DIR:$MLFLOW_DIR \
    $SIF_FILE \
    bash -c "
        set -e
        cd /code
        export DATA_DIR=$DATA_DIR
        export ROOT_DIR=$ROOT_DIR
        export WORK_DIR=$WORK_DIR
        export MLFLOW_DIR=$MLFLOW_DIR
        python train.py -cn apptainer_config_test.yaml  #test_meps_numpy_datamodule.py #
    "
```
