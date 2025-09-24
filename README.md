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
module load env/release/2024.1
module load Apptainer/1.3.6-GCCcore-13.3.0
module load git


DATA_DIR=/home/users/u101329/p200177_t2/DE_371/datasets/datasets_SMHI/npy_intep
ROOT_DIR=/home/users/u101329/p200177_t2/u101329/tests/exp-01
MLFLOW_DIR=/home/users/u101329/p200177_t2/u101329/_mlruns
SIF_FILE=../containers/container.sif 
apptainer exec \
    --nv \
    --containall \
    --bind .:/code \
    --bind $ROOT_DIR:/root_dir \
    --bind $DATA_DIR/:/data \
    --bind $MLFLOW_DIR:/_mlruns \
    $SIF_FILE \
    bash -c '
        set -e
        cd /code
        export DATA_DIR=/data
        export ROOT_DIR=/root_dir
        export WORK_DIR=$ROOT_DIR/_work
        export MLFLOW_DIR=/_mlruns
        [ -d $WORK_DIR ] || mkdir -p $WORK_DIR
        [ -d $MLFLOW_DIR ] || mkdir -p $MLFLOW_DIR
        python train.py
    '
```
