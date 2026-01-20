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

## Sample 

- read the example  [here](generate/README.md)
