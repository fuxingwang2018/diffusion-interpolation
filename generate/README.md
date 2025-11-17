# Generate samples
- With simple python
```bash

export DATA_DIR=/mnt/tier1/project/p200177/u101329/npy_interp
export OUT_DIR=/mnt/tier1/project/p200177/u101329/results/samples_shuff_2025_1 #

salloc -A p200177  -p gpu   --qos default -N 1 -t 10:00:00


 
- generate samples from the shufeled dataset trained on uvt only 

```bash
./generate/gen_2025_shuf-utv.sh  "[0,1,2,3]"   "cuda:0" &> gen_2025_shuf-utv.2.log&
./generate/gen_2025_shuf-utv.sh  "[4,5,6,7]"   "cuda:1" &> gen_2025_shuf-utv.2.log&
./generate/gen_2025_shuf-utv.sh  "[8,9,10,11]" "cuda:2" &> gen_2025_shuf-utv.3.log&
./generate/gen_2025_shuf-utv.sh  "[12,13,14]"  "cuda:3" &> gen_2025_shuf-utv.4.log&

``` 

- generate samples from the shufeled dataset trained on ruvt only 

```bash
./generate/gen_2025_shuf-ruvt.sh  "[0,1,2,3]"   "cuda:0" &> gen_2025_shuf-rutv.1.log&
./generate/gen_2025_shuf-ruvt.sh  "[4,5,6,7]"   "cuda:1" &> gen_2025_shuf-rutv.2.log&
./generate/gen_2025_shuf-ruvt.sh  "[8,9,10,11]" "cuda:2" &> gen_2025_shuf-rutv.3.log&
./generate/gen_2025_shuf-ruvt.sh  "[12,13,14]"  "cuda:3" &> gen_2025_shuf-rutv.4.log&


./generate/gen_2025_shuf-ruvt-100-steps.sh  "[0,1,2,3]"   "cuda:0" &> gen_2025_shuf-rutv-100-steps.1.log&

./generate/gen_2024_shuf-ruvt-40-steps.sh  "[0,1,2,3]"   "cuda:0" &> gen_2024_shuf-rutv-40-steps.1.log&
./generate/gen_2024_shuf-ruvt-40-steps.sh  "[4,5,6,7]"   "cuda:1" &> gen_2024_shuf-rutv-40-steps.2.log&
./generate/gen_2024_shuf-ruvt-40-steps.sh  "[8,9,10,11]" "cuda:2" &> gen_2024_shuf-rutv-40-steps.3.log&
./generate/gen_2024_shuf-ruvt-40-steps.sh  "[12,13,14]"  "cuda:3" &> gen_2024_shuf-rutv-40-steps.4.log&

``` 
 
nvi
 - generate samples from the shufeled dataset trained on ruvt only 

```bash
./generate/gen_2024-uvt.sh   "[0,1,2,3]"   "cuda:0" &> gen_2024-uvt.1.log&
./generate/gen_2024-uvt.sh   "[4,5,6,7]"   "cuda:1" &> gen_2024-uvt.2.log&
./generate/gen_2024-uvt.sh   "[8,9,10,11]" "cuda:2" &> gen_2024-uvt.3.log&
./generate/gen_2024-uvt.sh   "[12,13,14]"  "cuda:3" &> gen_2024-uvt.4.log&

``` 

```bash
./generate/gen_2024-uvt2.sh  "[2024-10-22,2024-10-31]"  "cuda:0" &> gen_2024-uvt2.1.log&
./generate/gen_2024-uvt2.sh  "[2024-11-01,2024-11-20]"  "cuda:1" &> gen_2024-uvt2.2.log&
./generate/gen_2024-uvt2.sh  "[2024-11-21,2024-12-10]"  "cuda:2" &> gen_2024-uvt2.3.log&
./generate/gen_2024-uvt2.sh  "[2024-12-11,2024-12-31]"  "cuda:3" &> gen_2024-uvt2.4.log&

``` 


```bash
./generate/gen_2025-uvt2.sh  "[2025-01-01,2025-01-31]"  "cuda:0" &> gen_2025-uvt2.1.log&
./generate/gen_2025-uvt2.sh  "[2025-02-01,2025-02-28]"  "cuda:1" &> gen_2025-uvt2.2.log&
./generate/gen_2025-uvt2.sh  "[2025-03-01,2025-03-15]"  "cuda:2" &> gen_2025-uvt2.3.log&
./generate/gen_2025-uvt2.sh  "[2025-03-16,2025-03-31]"  "cuda:3" &> gen_2025-uvt2.4.log&
``` 

```bash
./generate/gen_2025-uvt3.sh  "[2025-01-01,2025-01-31]"  "cuda:0" &> gen_2025-uvt3.1.log&
./generate/gen_2025-uvt3.sh  "[2025-02-01,2025-02-28]"  "cuda:1" &> gen_2025-uvt3.2.log&
./generate/gen_2025-uvt3.sh  "[2025-03-01,2025-03-15]"  "cuda:2" &> gen_2025-uvt3.3.log&
./generate/gen_2025-uvt3.sh  "[2025-03-16,2025-03-31]"  "cuda:3" &> gen_2025-uvt3.4.log&
``` 

```bash
./generate/gen_2024-uvt3.sh  "[2024-10-22,2024-10-31]"  "cuda:0" &> gen_2024-uvt3.1.log&
./generate/gen_2024-uvt3.sh  "[2024-11-01,2024-11-20]"  "cuda:1" &> gen_2024-uvt3.2.log&
./generate/gen_2024-uvt3.sh  "[2024-11-21,2024-12-10]"  "cuda:2" &> gen_2024-uvt3.3.log&
./generate/gen_2024-uvt3.sh  "[2024-12-11,2024-12-31]"  "cuda:3" &> gen_2024-uvt3.4.log&

``` 


```
-rw-r----- 1 u101329 p200177 367538631 Sep 29 20:23 _work/ddpm-npy-ruvt-20250929_134920/checkpoints/last.ckpt

_work/ddpm-npy-ruvt-20250929_151421/checkpoints/last.ckpt

```