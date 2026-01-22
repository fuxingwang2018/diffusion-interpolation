from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import xarray as xr
import lightning as L
from torch.utils.data import Dataset, DataLoader
from lightning.pytorch.utilities.rank_zero import rank_zero_info

import torch.nn as nn
 
 
 


# ============================================================
# Split Configuration
# ============================================================

@dataclass
class SplitConfig:
    """
    Split strategy configuration.
    
    strategy="sample":
      Cyclic allocation of samples:
      [train] -> [skip] -> [validation] -> [skip] -> [test] -> [skip] -> repeat...
      Values are COUNTS of samples (windows).
    """
    strategy: Literal["sample", "ratio", "none"] = "sample"
    
    # Counts for "sample" strategy
    train: int = 20
    validation: int = 2
    test: int = 2
    skip: int = 1

# ============================================================
# Normalizer (Optimized for GPU)
# ============================================================

class Normalizer(nn.Module):
    """
    Torch Module for normalization. 
    Stores stats as buffers to allow seamless move to GPU with the model.
    """
    def __init__(
        self,
        mode: Literal["none", "zscore", "symrange"],
        mean: Optional[torch.Tensor] = None,
        std: Optional[torch.Tensor] = None,
        minimum: Optional[torch.Tensor] = None,
        maximum: Optional[torch.Tensor] = None,
        norm_const:  Optional[torch.Tensor] = None,
    ) -> None:
        super().__init__()
        self.mode = mode
        
        
        if mode == "none":
            return

        # Register buffers so they move to device (GPU) with the module
        if mode == "zscore":
            assert mean is not None and std is not None, "zscore normalization requires mean and std"
            self.register_buffer("mean", mean.view(-1, 1)
            self.register_buffer("std", std.view(-1, 1))
            # Avoid div by zero
            self.std[self.std == 0] = 1.0
            
        elif mode == "symrange":
            assert mean is not None and minimum is not None and maximum is not None and norm_const is not None, \
                "symrange normalization requires mean, min, max, and norm_const"
            self.register_buffer("mean", mean.view(-1, 1))
            self.register_buffer("minimum",  maximum.view(-1, 1))
            self.register_buffer("maximum", minimum.view(-1, 1))
            self.register_buffer("norm_const", norm_const.view(-1, 1))
            # Pre-calculate denominator to save compute during forward
            denom = torch.maximum(torch.abs(self.minimum - self.mean), torch.abs(self.maximum - self.mean))
            denom[denom == 0] = 1.0
            self.register_buffer("denom", denom)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input x: (..., C, Cell) or (..., C, H, W)
        """
        if self.mode == "none":
            return x
        
        # PyTorch broadcasting handles the shapes automatically
        # x shape: (Batch, Time, C, Cell)
        # stat shape: (C, 1) -> broadcasts to (1, 1, C, 1) effectively
        
        if self.mode == "zscore":
            return (x - self.mean) / self.std
            
        if self.mode == "symrange":
            return self.norm_const * (x - self.mean) / self.denom
            
        return x
    
    def denormalize(self, x: torch.Tensor) -> torch.Tensor:
        """Inverse operation for metrics/visualization"""
        if self.mode == "none":
            return x
            
        if self.mode == "zscore":
            return (x * self.std) + self.mean
            
        if self.mode == "symrange":
            return (x * self.denom / self.norm_const) + self.mean
            
        return x


# ============================================================
# Dataset (Lean & Fast)
# ============================================================

class MEPSZarrDataset(Dataset):
    """
    Dataset returns RAW data tensors. 
    Normalization should be applied by the calling loop/model on GPU.
    """
    def __init__(
        self,
        dataset_path: str,
        indices: Sequence[int],
        window_size: int = 7,
        variable_indices: Sequence[int] = (0, 1, 2, 3),
        dtype: Literal["float32", "float16"] = "float32",
    ) -> None:
        super().__init__()
        self.indices = indices 
        self.window_size = window_size
        self.var_idx = list(variable_indices)
        
        self.dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        self.out_dtype = self.dtype_map[dtype]
        
        # Lazy open with xarray
        self.ds = xr.open_dataset(dataset_path, engine='zarr', chunks='auto')
        self.data_var = self.ds['data'] # Keep direct reference for speed

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int):
        # 1. Determine time slice
        start_t = self.indices[i]
        end_t = start_t + self.window_size
        
        # 2. Load Data: (Time, Var, Ens, Cell)
        # Using slice is more efficient than individual indices
        # .values converts to numpy immediately
        data_block = self.data_var.isel(
            time=slice(start_t, end_t),
            variable=self.var_idx,
            ensemble=0 
        ).values # Shape: (Window, C, Cell)
        
        # 3. Create Tensors directly (No CPU Normalization loop)
        # We split Boundary (x) and Internal (y)
        # x: First and Last frame
        # y: Intermediate frames
        
        if self.window_size < 3:
             raise ValueError(f"Window size must be >= 3, got {self.window_size}")

        # Construct x and y
        # Note: np.stack creates a copy, which is fine here.
        x_np = np.stack([data_block[0], data_block[-1]], axis=0)
        y_np = data_block[1:-1]

        # Convert to tensor once
        x_t = torch.as_tensor(x_np, dtype=self.out_dtype)
        y_t = torch.as_tensor(y_np, dtype=self.out_dtype)
        
        # Metadata
        meta = {
            "index": i,
            "start_time_idx": int(start_t),
            "end_time_idx": int(end_t),
            # Accessing dates can be slow, handle gracefully
            "date": str(self.ds.dates.values[start_t]) if 'dates' in self.ds else ""
        }

        return x_t, y_t, meta

# ============================================================
# Lightning DataModule
# ============================================================

class MEPSZarrDataModule(L.LightningDataModule):
    def __init__(
        self,
        dataset_path: str,
        # Data Params
        window_size: int = 7,
        variables: Optional[Sequence[str]] = None,
        # Normalization
        normalize: Literal["none", "zscore", "symrange"] = "symrange",
        norm_const: float = 0.95,
        # Dtype
        dtype: Literal["float32", "float16"] = "float32",
        # Loader
        batch_size: int = 32,
        num_workers: int = 2,
        pin_memory: bool = True,
        shuffle_train: bool = True,
        # Split Config
        split: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        
        self.dataset_path = dataset_path
        self.window_size = window_size
        self.variables = list(variables) if variables is not None else None
        self.dtype = dtype
        
        # Stride = Window Size - 1 (Overlap of 1 frame)
        # e.g., if W=7: S=6. 
        # Sample 0: [0,1,2,3,4,5,6]
        # Sample 1: [6,7,8,9,10,11,12]
        assert window_size >= 3, "Window size must be at least 3"
        self.stride = window_size - 1 

        self.normalize_mode = normalize
        self.norm_const = norm_const
        
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.shuffle_train = shuffle_train
        
        self.split_cfg = SplitConfig(**split) if split is not None else SplitConfig()

        self.train_ds: Optional[Dataset] = None
        self.val_ds: Optional[Dataset] = None
        self.test_ds: Optional[Dataset] = None
        self.var_indices: List[int] = [] 

    def _get_split_indices(self, total_time_steps: int) -> Tuple[List[int], List[int], List[int]]:
            """
            Assigns valid start indices using the 'sample' strategy (cyclic counts).
            Pattern: [train] -> [skip] -> [validation] -> [skip] -> [test] -> [skip] -> repeat...
            """
            cfg = self.split_cfg
            
            # 1. Calculate ALL valid start indices for sliding windows
            # Last valid start is T - window_size
            max_start = total_time_steps - self.window_size
            if max_start < 0:
                raise ValueError(f"Total time ({total_time_steps}) is shorter than window size ({self.window_size})")

            all_starts = np.arange(0, max_start + 1, self.stride)
            
            if cfg.strategy != "sample":
                # Fallback or other strategies
                rank_zero_info(f"Split strategy '{cfg.strategy}' not explicitly handled, will exit")
                exit(1)

            # 2. Cyclic Allocation
            train_idx, val_idx, test_idx = [], [], []
            
            total_samples = len(all_starts)
            idx_ptr = 0
            
            # Cache counts for clarity
            n_train = int(cfg.train)
            n_val = int(cfg.validation)
            n_test = int(cfg.test)
            n_skip = int(cfg.skip)
            
            # Loop until we run out of samples
            while idx_ptr < total_samples:
                # --- 1. Train Block ---
                end = min(idx_ptr + n_train, total_samples)
                train_idx.extend(all_starts[idx_ptr:end])
                idx_ptr = end
                
                # Skip after Train (Train -> Skip -> Val)
                idx_ptr += n_skip
                if idx_ptr >= total_samples: break
                
                # --- 2. Validation Block ---
                end = min(idx_ptr + n_val, total_samples)
                val_idx.extend(all_starts[idx_ptr:end])
                idx_ptr = end
                
                # Skip after Validation (Val -> Skip -> Test)
                idx_ptr += n_skip
                if idx_ptr >= total_samples: break
                
                # --- 3. Test Block ---
                end = min(idx_ptr + n_test, total_samples)
                test_idx.extend(all_starts[idx_ptr:end])
                idx_ptr = end
                
                # Skip after Test (Test -> Skip -> Train/Repeat)
                idx_ptr += n_skip
                    
            return train_idx, val_idx, test_idx
    def setup(self, stage: Optional[str] = None) -> None:
        # Open Zarr to read Metadata
        ds = xr.open_dataset(self.dataset_path, engine='zarr')
        total_time = ds.sizes['time']
        
        # 1. Resolve Variable Names to Indices
        all_var_names = list(ds['variable'].values)
        if self.variables is None:
            self.var_indices = list(range(len(all_var_names)))
        else:
            try:
                self.var_indices = [all_var_names.index(str(v)) for v in self.variables]
            except ValueError as e:
                raise ValueError(f"Requested variable not found in Zarr. Available: {all_var_names}. Error: {e}")

        rank_zero_info(f"Selected variables: {self.variables} -> Indices: {self.var_indices}")

        # 2. Prepare Normalizer
        stats_dict = {}
        if self.normalize_mode != "none":
            try:
                stats_dict['mean'] = ds['mean'].values 
                stats_dict['std'] = ds['stdev'].values
                stats_dict['mean'] = ds['mean'].values 
                stats_dict['minimum'] = ds['minimum'].values
                stats_dict['maximum'] = ds['maximum'].values
            except KeyError as e:
                rank_zero_info(f"Warning: Could not load stat {e} from Zarr. Normalization might fail.")

        self.normalizer = Normalizer(
            mode=self.normalize_mode,
            stats_dict=stats_dict,
            norm_const=self.norm_const,
            channel_indices=self.var_indices 
        )

        # 3. Split Indices
        train_idx, val_idx, test_idx = self._get_split_indices(total_time)
        
        rank_zero_info(f"Total valid windows: {len(train_idx)+len(val_idx)+len(test_idx)}")
        rank_zero_info(f"Train samples: {len(train_idx)}")
        rank_zero_info(f"Val samples:   {len(val_idx)}")
        rank_zero_info(f"Test samples:  {len(test_idx)}")

        # 4. Create Datasets
        self.train_ds = MEPSZarrDataset(
            self.dataset_path, train_idx, self.window_size, self.var_indices, self.normalizer, self.dtype
        )
        self.val_ds = MEPSZarrDataset(
            self.dataset_path, val_idx, self.window_size, self.var_indices, self.normalizer, self.dtype
        )
        self.test_ds = MEPSZarrDataset(
            self.dataset_path, test_idx, self.window_size, self.var_indices, self.normalizer, self.dtype
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=self.shuffle_train, 
                          num_workers=self.num_workers, pin_memory=self.pin_memory)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_ds, batch_size=self.batch_size, shuffle=False, 
                          num_workers=self.num_workers, pin_memory=self.pin_memory)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.test_ds, batch_size=self.batch_size, shuffle=False, 
                          num_workers=self.num_workers, pin_memory=self.pin_memory)