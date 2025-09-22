# MEPSNPYDataModule — Options & Usage Guide

This documents every option in the **MEPSNPYDataModule** and the shapes you should expect under different modes. It assumes you are using the grouped CSV produced by your sequence finder (with JSON columns `Members`, `LeadTimes`, `MemberPaths`).

---

## 1) Input sources

### `root: str`

Filesystem root prepended to relative paths in the CSV.

* Example: `root: /data`

### `sequences_csv: str | null`

Single **grouped** CSV file to read, then split into train/val/test according to `split`.

* Mutually exclusive with the `{train,val,test}_csv` trio.

### `train_csv`, `val_csv`, `test_csv: str | null`

Use **separate grouped CSVs**; no internal splitting is done.

* If any of these is set, `sequences_csv` is ignored.

### CSV schema (grouped)

Each row represents one `(Date, Window)` with all members’ files aligned by lead time:

* `Members`: JSON list of member IDs, e.g., `[0,1,2]`
* `LeadTimes`: JSON list of lead times, e.g., `[0,1,2,3,4,5,6]`
* `MemberPaths`: JSON 2-D list `[[m0_lt0, m0_lt1, ...], [m1_lt0, m1_lt1, ...], ...]`

The DataModule converts relative paths to absolute by joining with `root`.

---

## 2) Filtering & inclusion

### `members: list[int] | null`

Only keep these members. If `null`, use all members present in each row.

* Example: `members: [0,2]`

### `leads: list[int] | null`

Only keep these lead times. If `null`, use all lead times present in the row.

### `require_internal_targets: bool`

When `true`, require at least **3** lead times (so there’s an internal target), otherwise rows with `<3` leads are dropped.

### `require_all_selected_members: bool`

When you set `members`, dropping any row where **any** requested member is missing keeps the member count `M` consistent across samples (handy for fixed shapes).

* Set to `false` if you prefer to keep partial member-sets.

---

## 3) Sample building & shapes

Let:

* `C = len(file_channel_indices)` (channels per `.npy` file you select from the 4 available)
* `T = number of lead times` **after** filtering/sorting
* `M = number of members` **after** filtering
* Endpoint logic: **Inputs** use the **first & last** lead per member; **Targets** use **all internal** leads (everything between first and last).

### `sample_mode: "ensemble" | "per_member"`

* **`ensemble`** → One sample per `(Date, Window)` that **includes all members** (member axis kept).
* **`per_member`** → One sample per `(Date, Window, Member)` (member axis removed).

### `file_channel_indices: list[int]`

Which channels from each `(4, H, W)` file to read.

* Example: `[0,1,2]` → `C = 3`.

### `stack_time_on_channel: bool`

If `true`, the time axis (endpoints or internals) is **flattened into channels**.
If `false`, a true time axis is kept.

### `stack_member_on_channel: bool` (only applies to `sample_mode: "ensemble"`)

If `true`, the member axis is also **flattened into channels** (useful for plain CNNs that only accept `(B,C,H,W)`).

---

### Shape summary

* **`sample_mode: ensemble`** (keeps all members in a sample)

  If `stack_time_on_channel: true`, `stack_member_on_channel: false` (default):

  * `x`: `(B, M, 2*C, H, W)` — endpoints for each member
  * `y`: `(B, M, (T-2)*C, H, W)` — internals for each member

  If `stack_time_on_channel: false`, `stack_member_on_channel: false`:

  * `x`: `(B, M, 2, C, H, W)`
  * `y`: `(B, M, T-2, C, H, W)`

  If `stack_time_on_channel: true`, `stack_member_on_channel: true` (fully fused for vanilla CNNs):

  * `x`: `(B, M*2*C, H, W)`
  * `y`: `(B, (T-2)*M*C, H, W)`

* **`sample_mode: per_member`** (one member per sample)

  If `stack_time_on_channel: true`:

  * `x`: `(B, 2*C, H, W)`
  * `y`: `(B, (T-2)*C, H, W)`

  If `stack_time_on_channel: false`:

  * `x`: `(B, 2, C, H, W)`
  * `y`: `(B, T-2, C, H, W)`

**Model tip:**

* With fully fused `(B,C,H,W)`, set `model.in_channels` to the `C` shown by your chosen mode.
* Otherwise (member/time axes present), either adapt the model to accept those extra dims or fuse inside the model.

---

## 4) Normalization

### `normalize: "none" | "zscore" | "symrange"`

* `"none"` → No normalization.
* `"zscore"` → `(x - mean) / std`
* `"symrange"` → `norm_const * (x - average) / max(|global_min - average|, |global_max - average|)`

### `stats_npz: str | null`

Path to `.npz` file containing stats. Required for `"zscore"` and `"symrange"`.

### Key names in the `.npz` (configurable)

* For `"zscore"`:

  * `mean_key` (default: `"mean"`)
  * `std_key`  (default: `"std"`)
* For `"symrange"`:

  * `average_key` (default: `"average"`)
  * `global_min_key` (default: `"global_min"`)
  * `global_max_key` (default: `"global_max"`)
  * `norm_const: float` (default: `1.0`)

**Broadcasting:** The stats arrays may be scalar, `(C,)`, `(H,W)`, or `(C,H,W)` — they are broadcast to your data shape `(C,H,W)`. If broadcasting is impossible, you’ll get a clear error.

**Safety:** Denominators of `0` are internally replaced with `1.0` to avoid division by zero.

---

## 5) DataLoader & performance

### `batch_size: int`

Batch size used by each loader.

### `num_workers: int`

PyTorch dataloader workers. Set `>0` for parallel file reads.

### `pin_memory: bool`

Enable pinned memory (useful on GPUs).

### `persistent_workers: bool | null`

If `null`, it’s derived as `num_workers > 0`. Keeping workers alive between epochs often speeds up training.

### `shuffle_train: bool`

Shuffles the training set each epoch.

### `dtype: "float32" | "float16" | "bfloat16" | "float64"`

Torch dtype of returned tensors.

### `mmap: bool`

Use `np.load(..., mmap_mode="r")` for efficient, read-only memory mapping of `.npy` files (recommended for large datasets).

---

## 6) Splitting

### `split.type: "ratio" | "count" | "none"`

* `ratio`: use `train`, `val`, and the remainder for `test`.
* `count`: absolute counts for `train`, `val`, `test`.
* `none`: everything goes to train (val/test empty).

### `split.train`, `split.val`, `split.test`

* When `type="ratio"`, they are fractions (e.g., `0.8`, `0.1`, `0.1`).
* When `type="count"`, they are integers that must sum to ≤ dataset size.

### `split.seed`, `split.shuffle_before_split`

Control deterministic shuffling before splitting.

---

## 7) End-to-end examples

### A) **Ensemble sample** with member axis kept

```yaml
datamodule:
  _target_: data.meps_npy_datamodule.MEPSNPYDataModule
  root: /samples
  sequences_csv: /data/sequences-test.csv
  members: [0,1,2]                # keep 3 members
  leads: null                     # all leads present
  require_internal_targets: true
  require_all_selected_members: true

  sample_mode: ensemble
  file_channel_indices: [0, 1, 2] # C=3
  stack_time_on_channel: true     # fuse time → channels
  stack_member_on_channel: false  # keep member dim

  normalize: symrange
  stats_npz: /data/sequences-stats-test.npz
  average_key: global_average
  global_min_key: global_min
  global_max_key: global_max
  norm_const: 0.95

  batch_size: 8
  num_workers: 4
  mmap: true

  split:
    type: ratio
    train: 0.8
    val: 0.1
    test: 0.1
    seed: 42
    shuffle_before_split: false
```

**Shapes:** `x: (B, M, 2*C, H, W) = (B, 3, 6, H, W)`, `y: (B, 3, (T-2)*3, H, W)`.

### B) **Per-member sample** (one member per sample)

```yaml
datamodule:
  _target_: data.meps_npy_datamodule.MEPSNPYDataModule
  root: /samples
  sequences_csv: /data/sequences-test.csv
  members: null
  leads: [0,1,2,3,4,5,6]         # require_internal_targets=true → ok (T≥3)

  sample_mode: per_member
  file_channel_indices: [0, 1, 2]
  stack_time_on_channel: true

  normalize: none
  batch_size: 32
  num_workers: 2
  mmap: true
```

**Shapes:** `x: (B, 2*C, H, W) = (B, 6, H, W)`, `y: (B, (T-2)*C, H, W)`.

### C) **Fully fused for CNNs** (ensemble + fuse members and time)

```yaml
datamodule:
  _target_: data.meps_npy_datamodule.MEPSNPYDataModule
  root: /samples
  sequences_csv: /data/sequences-test.csv
  members: [0,1,2,3]
  sample_mode: ensemble
  file_channel_indices: [0,1]
  stack_time_on_channel: true
  stack_member_on_channel: true
  normalize: symrange
  stats_npz: /data/stats.npz
```

**Shapes:** `x: (B, M*2*C, H, W) = (B, 4*2*2, H, W) = (B, 16, H, W)`.

**Model hint:** set `model.in_channels` to the resulting channel count (here, `16`).

---

## 8) Sanity-check snippet (works in DDP; prints only on rank 0)

```python
from hydra.utils import instantiate
from lightning.pytorch.utilities import rank_zero_only

dm = instantiate(cfg.datamodule, _recursive_=False)
dm.setup("fit")

@rank_zero_only
def peek(dm):
    xb, yb = next(iter(dm.train_dataloader()))
    print("x:", tuple(xb.shape), "y:", tuple(yb.shape))

peek(dm)
```

---

## 9) Error messages & common pitfalls

* **“requires stats\_npz”**: You set `normalize: zscore` or `symrange` without `stats_npz`.
* **“Stats shape … not broadcastable …”**: Your stats arrays can’t broadcast to `(C,H,W)`. Use scalar, `(C,)`, `(H,W)`, or `(C,H,W)`.
* **Dropped rows**: With `require_internal_targets: true`, any row with `<3` leads is dropped; with `require_all_selected_members: true`, rows missing any requested member are also dropped. If your dataset becomes empty, relax these flags.
* **Model in\_channels mismatch**: Recalculate from the chosen mode
  (`2*C`, `(T-2)*C`, `M*2*C`, etc.) and update the model accordingly.

---

## 10) DDP notes

* Stateless reads and `np.load(..., mmap_mode='r')` are **DDP-safe**.
* Lightning auto-injects `DistributedSampler`; keep `shuffle_train: true` for train only.
* Use `L.seed_everything(seed, workers=True)` to keep randomness deterministic across ranks/workers.

---

## References (docs)

```text
PyTorch Lightning DataModule: https://lightning.ai/docs/pytorch/stable/data/datamodule.html
PyTorch DataLoader:           https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader
NumPy memmap:                 https://numpy.org/doc/stable/reference/generated/numpy.load.html
Hydra instantiation:          https://hydra.cc/docs/advanced/instantiate_objects/overview/
```

If you want, I can generate a **ready-to-run minimal config group** (e.g., `conf/preset/…`) so you can switch between the three presets with `preset=<name>` from the CLI.
