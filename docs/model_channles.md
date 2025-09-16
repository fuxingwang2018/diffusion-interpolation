# How to compute channel counts

Let

* `C = len(datamodule.file_channel_indices)` (channels per file you load, from the 4 available)
* `T = number of lead times kept` **after** `datamodule.leads` filtering (e.g. `[0..6]` → `T=7`)
* `M = number of members kept` **after** `datamodule.members` filtering

And remember the DM builds:

* **inputs** = endpoints = **first + last** lead(s)
* **targets** = internals = **all leads between** first and last → count = `T-2`

> To keep shapes constant across the whole dataset, it’s best to set both `members:` and `leads:` in the config and also keep
> `require_all_selected_members: true` and `require_internal_targets: true`.

## Case A — `sample_mode: per_member`

* If `stack_time_on_channel: true` (recommended for the current EDM model):

  * **cond\_channels** = `2 * C`
  * **target\_channels** = `(T - 2) * C`
* If `stack_time_on_channel: false`: the DM returns a **time dimension**, not channels. The provided EDM model expects `(B, C, H, W)`, so either:

  * set `stack_time_on_channel: true`, **or**
  * change the model to handle a time axis (not covered here).

## Case B — `sample_mode: ensemble`

* If `stack_member_on_channel: true` **and** `stack_time_on_channel: true` (recommended for vanilla CNN/UNet):

  * **cond\_channels** = `M * 2 * C`
  * **target\_channels** = `M * (T - 2) * C`
* If `stack_member_on_channel: false`: the DM keeps an **M dimension**. The provided EDM model expects `(B, C, H, W)`, so either:

  * set `stack_member_on_channel: true`, **or**
  * extend the model to accept a member axis.

# Quick examples

### Example 1 — per\_member, 3 vars, 7 leads

```yaml
datamodule:
  sample_mode: per_member
  file_channel_indices: [0,1,2]    # C=3
  leads: [0,1,2,3,4,5,6]           # T=7
  stack_time_on_channel: true

model:
  _target_: models.edm_interpolator.EDMInterpolator
  cond_channels: 6                 # 2*C = 2*3
  target_channels: 15              # (T-2)*C = 5*3
```

### Example 2 — ensemble (fused), 3 members, 2 vars, 7 leads

```yaml
datamodule:
  sample_mode: ensemble
  members: [0,1,2]                 # M=3
  file_channel_indices: [0,1]      # C=2
  leads: [0,1,2,3,4,5,6]           # T=7
  stack_time_on_channel: true
  stack_member_on_channel: true    # fuse members → channels

model:
  _target_: models.edm_interpolator.EDMInterpolator
  cond_channels: 12                # M*2*C = 3*2*2
  target_channels: 30              # M*(T-2)*C = 3*5*2
```

# Guardrail (optional but recommended)

Even if you set the numbers manually, add a tiny **sanity check** so a mismatch fails fast:

```python
# after dm.setup("fit")
x0, y0 = dm.train_set[0]

def fuse_member_if_any(t):
    return t if t.ndim == 3 else t.reshape(t.shape[0]*t.shape[1], *t.shape[-2:])  # (M,C,H,W)->(M*C,H,W)

def channels_of(t):
    t = fuse_member_if_any(t)
    assert t.ndim == 3, f"Expected (C,H,W) after fusing, got {t.shape}"
    return t.shape[0]

inferred_cond = channels_of(x0)
inferred_targ = channels_of(y0)

assert inferred_cond == cfg.model.cond_channels, \
    f"cond_channels mismatch: cfg={cfg.model.cond_channels}, dataset={inferred_cond}"
assert inferred_targ == cfg.model.target_channels, \
    f"target_channels mismatch: cfg={cfg.model.target_channels}, dataset={inferred_targ}"
```

# Tips & pitfalls

* If you see shape errors, re-check `C`, `T`, `M`. The most common mistake is counting the wrong `T` (remember targets use **T−2** leads).
* Keep `require_all_selected_members: true` so `M` is constant per sample in **ensemble** mode.
* The provided EDM model assumes `(B, C, H, W)`. Use `stack_time_on_channel: true` and, for ensemble, `stack_member_on_channel: true` unless you’ve modified the model to accept extra dims.

That’s it—set the two integers in `model:` using the formulas above, and you’re good.
