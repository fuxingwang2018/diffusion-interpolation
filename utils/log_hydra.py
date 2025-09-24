# after `logger = instantiate(cfg.logger)` in train.py
from lightning.pytorch.utilities import rank_zero_only
from omegaconf import OmegaConf
from pathlib import Path
import tempfile, os

@rank_zero_only
def log_cfg_to_mlflow(logger, cfg):
    # Print for stdout logs
    txt = OmegaConf.to_yaml(cfg, resolve=True)
 
    run_dir = Path(cfg._work_dir) 
    with tempfile.NamedTemporaryFile(mode="w", dir=run_dir, prefix="config_resolved_", suffix=".yaml", delete=False) as f:
        f.write(txt)
        tmp_path = Path(f.name)

    # Upload as an artifact under "hydra/config_resolved.yaml"
    logger.experiment.log_artifact(
        logger.run_id, str(tmp_path), artifact_path="hydra"
    )

    # Clean up the temp file after upload
    try:
        tmp_path.unlink()
    except OSError:
        pass

 