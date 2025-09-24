sweet — here are **ready-to-use Hydra model configs** that cover the main variants:

* EDM (baseline)
* EDM + SDEdit
* DDPM (predict ε)
* DDPM (predict v / implicit)
* DDPM + SDEdit (works with either ε or v; shown with ε)
* DDIM sampling flavor for DDPM (optional `eta`/`steps` controls)

> Note: targets match the multi-file code:
> `models.edm.EDMInterpolator` and `models.ddpm.DDPMInterpolator`.

## File Structure

```
diffusion-interp/
└── conf
    ├── model
    │   ├── edm_interpolator.yaml
    │   ├── edm_interpolator_sdedit.yaml
    │   ├── ddpm_eps.yaml
    │   ├── ddpm_v.yaml
    │   ├── ddpm_sdedit.yaml
    │   └── ddpm_ddim.yaml

```

### 1) EDM (baseline) — `conf/model/edm_interpolator.yaml`

```yaml
_target_: models.edm.EDMInterpolator

# channels
cond_channels: 6
target_channels: 15

# UNet
unet_base: 64
time_embed_dim: 256

# EDM noise / training
sigma_data: 0.5
sigma_min: 0.002
sigma_max: 80.0
p_mean: -1.2
p_std: 1.2
loss_weighting: edm  # or 'none'

# sampling
sample_steps: 30
rho: 7.0
sample_every_val: 1
sample_save_npz: true

# SDEdit (off)
sdedit_enabled: false
sdedit_sigma: null

# optimizer / scheduler
optimizer_cfg:
  _target_: torch.optim.AdamW
  lr: 1.0e-4
  weight_decay: 1.0e-4
scheduler_cfg:
  _target_: torch.optim.lr_scheduler.ExponentialLR
  gamma: 0.9999

# figures / files
figures_cfg:
  root: "_work/figs"
  use_meta_subdirs: true
  filename_template: "{date}_{window}_m{member}"
```

---

### 2) EDM + SDEdit — `conf/model/edm_interpolator_sdedit.yaml`

```yaml
_target_: models.edm.EDMInterpolator

# channels
cond_channels: 6
target_channels: 15

# UNet
unet_base: 64
time_embed_dim: 256

# EDM noise / training
sigma_data: 0.5
sigma_min: 0.002
sigma_max: 80.0
p_mean: -1.2
p_std: 1.2
loss_weighting: edm

# sampling
sample_steps: 30
rho: 7.0
sample_every_val: 1
sample_save_npz: true

# SDEdit (on)
sdedit_enabled: true
# If null, will use schedule midpoint; set a float to force a specific σ (e.g., 10.0)
sdedit_sigma: null

# optimizer / scheduler
optimizer_cfg:
  _target_: torch.optim.AdamW
  lr: 1.0e-4
  weight_decay: 1.0e-4
scheduler_cfg:
  _target_: torch.optim.lr_scheduler.ExponentialLR
  gamma: 0.9999

figures_cfg:
  root: "_work/figs"
  use_meta_subdirs: true
```

---

### 3) DDPM (ε-prediction) — `conf/model/ddpm_eps.yaml`

```yaml
_target_: models.ddpm.DDPMInterpolator

# channels
cond_channels: 6
target_channels: 15

# UNet
unet_base: 64
time_embed_dim: 256

# DDPM core
T: 1000
schedule: cosine   # or 'linear'
predict: eps       # ε-prediction; use 'v' for implicit model
loss_weighting: none

# validation sampling & saving
sample_every_val: 1
sample_save_npz: true

# SDEdit (off)
sdedit_enabled: false
sdedit_start_t: null
sdedit_start_pct: null

# optimizer / scheduler
optimizer_cfg:
  _target_: torch.optim.AdamW
  lr: 1.0e-4
  weight_decay: 1.0e-4
scheduler_cfg:
  _target_: torch.optim.lr_scheduler.ExponentialLR
  gamma: 0.9999

figures_cfg:
  root: "_work/figs"
  use_meta_subdirs: true
  filename_template: "{date}_{window}_m{member}"
```

---

### 4) DDPM (v-prediction / implicit) — `conf/model/ddpm_v.yaml`

```yaml
_target_: models.ddpm.DDPMInterpolator

# channels
cond_channels: 6
target_channels: 15

#  UNet
unet_base: 64
time_embed_dim: 256

# DDPM
T: 1000
schedule: cosine
predict: v          # v-prediction (implicit)
loss_weighting: none

# val sampling & saving
sample_every_val: 1
sample_save_npz: true

# SDEdit (off)
sdedit_enabled: false
sdedit_start_t: null
sdedit_start_pct: null

# optimizer / scheduler
optimizer_cfg:
  _target_: torch.optim.AdamW
  lr: 1.0e-4
  weight_decay: 1.0e-4
scheduler_cfg:
  _target_: torch.optim.lr_scheduler.ExponentialLR
  gamma: 0.9999

figures_cfg:
  root: "_work/figs"
  use_meta_subdirs: true
```

---

### 5) DDPM + SDEdit (ε-prediction) — `conf/model/ddpm_sdedit.yaml`

```yaml
_target_: models.ddpm.DDPMInterpolator

# channels
cond_channels: 6
target_channels: 15

# UNet

unet_base: 64
time_embed_dim: 256

# DDPM
T: 1000
schedule: cosine
predict: eps
loss_weighting: none

# SDEdit (on): choose start by t index or percent of T
sdedit_enabled: true
sdedit_start_t: null          # e.g., 400
sdedit_start_pct: 0.5         # 50% of T (used if start_t is null)

# val sampling & saving
sample_every_val: 1
sample_save_npz: true

# optimizer / scheduler
optimizer_cfg:
  _target_: torch.optim.AdamW
  lr: 1.0e-4
  weight_decay: 1.0e-4
scheduler_cfg:
  _target_: torch.optim.lr_scheduler.ExponentialLR
  gamma: 0.9999

figures_cfg:
  root: "_work/figs"
  use_meta_subdirs: true
```

---

### 6) DDPM with DDIM sampling flavor — `conf/model/ddpm_ddim.yaml`

```yaml
_target_: models.ddpm.DDPMInterpolator

# channels
cond_channels: 6
target_channels: 15

# & UNet
 
unet_base: 64
time_embed_dim: 256

# DDPM training settings
T: 1000
schedule: cosine
predict: eps
loss_weighting: none

# (Sampling is chosen at runtime via model.sample_from_cond(..., sampler="ddim", eta=..., steps=...))
# Keep SDEdit off here; can enable in a separate config if needed.
sdedit_enabled: false
sdedit_start_t: null
sdedit_start_pct: null

# val sampling & saving
sample_every_val: 1
sample_save_npz: true

# optimizer / scheduler
optimizer_cfg:
  _target_: torch.optim.AdamW
  lr: 1.0e-4
  weight_decay: 1.0e-4
scheduler_cfg:
  _target_: torch.optim.lr_scheduler.ExponentialLR
  gamma: 0.9999

figures_cfg:
  root: "_work/figs"
  use_meta_subdirs: true
```

---

### How to pick a config

* Baseline EDM: `+model=edm_interpolator`
* EDM with SDEdit: `+model=edm_interpolator_sdedit`
* DDPM (ε): `+model=ddpm_eps`
* DDPM (v / implicit): `+model=ddpm_v`
* DDPM with SDEdit (ε): `+model=ddpm_sdedit`
* DDPM trained model, use DDIM sampler at inference: `+model=ddpm_ddim` and call

  ```python
  model.sample_from_cond(x_cond, shape_target=(C,H,W), sampler="ddim", eta=0.0, steps=50)
  ```

If you want fewer files, you can keep only your preferred default (e.g., `edm_interpolator.yaml`) and swap others in as needed.
