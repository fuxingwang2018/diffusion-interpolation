from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple, Union
 

import numpy as np

import xarray as xr
import lightning as L
from torch.utils.data import Dataset, DataLoader
from lightning.pytorch.utilities.rank_zero import rank_zero_info
import torch
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
    def __init__(
        self,
        mode: Literal["none", "zscore", "symrange"],
        mean: Optional[torch.Tensor] = None,
        std: Optional[torch.Tensor] = None,
        minimum: Optional[torch.Tensor] = None,
        maximum: Optional[torch.Tensor] = None,
        norm_const: Optional[float] = None,
    ) -> None:
        super().__init__()
        self.mode = mode

        if mode == "none":
            self.register_buffer("dummy", torch.tensor(0.))
            return

        # Helper to ensure 1D shape (C,)
        def to_1d(t):
            if t is None:
                return None
            if not torch.is_tensor(t):
                t = torch.as_tensor(t)
            return t.view(-1)

        mean = to_1d(mean)
        std = to_1d(std)
        minimum = to_1d(minimum)
        maximum = to_1d(maximum)

        if mode == "zscore":
            assert mean is not None and std is not None
            self.register_buffer("mean", mean)
            # Avoid div by zero
            std = torch.where(std == 0, torch.ones_like(std), std)
            self.register_buffer("std", std)

        elif mode == "symrange":
            assert mean is not None and minimum is not None and maximum is not None and norm_const is not None
            self.register_buffer("mean", mean)
            self.register_buffer("minimum", minimum)
            self.register_buffer("maximum", maximum)
            self.register_buffer("norm_const", torch.tensor(norm_const))
            
            # Calculate denom (keep it 1D)
            denom = torch.maximum(torch.abs(self.minimum - self.mean), torch.abs(self.maximum - self.mean))
            denom = torch.where(denom == 0, torch.ones_like(denom), denom)
            self.register_buffer("denom", denom)

    def _get_stats_view(self, x: torch.Tensor, stat: torch.Tensor) -> torch.Tensor:
        """
        Reshapes stat (C,) to (1, ..., 1, C, 1, ..., 1) to match x.
        Handling:
        (..., C, Cell) -> view(..., C, 1)
        (..., C, H, W) -> view(..., C, 1, 1)
        """
        # Determine number of spatial dims.
        # Heuristic: Check if the 3rd-to-last dim matches C (implies Image: ..., C, H, W)
        # Otherwise assume Sequence (..., C, Cell)
        
        C = stat.shape[0]
        
        # Case: (..., C, H, W) -> C is at index -3
        if x.ndim >= 3 and x.shape[-3] == C:
            return stat.view(1, -1, 1, 1)
             
        # Case: (..., C, Cell) -> C is at index -2
        # Default fall-through for safety
        return stat.view(1, -1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.mode == "none":
            return x

        # Reshape stats to align with x
        mean = self._get_stats_view(x, self.mean)

        if self.mode == "zscore":
            std = self._get_stats_view(x, self.std)
            return (x - mean) / std

        if self.mode == "symrange":
            denom = self._get_stats_view(x, self.denom)
            return self.norm_const * (x - mean) / denom

        return x

    def denormalize(self, x: torch.Tensor) -> torch.Tensor:
        if self.mode == "none":
            return x
            
        mean = self._get_stats_view(x, self.mean)
            
        if self.mode == "zscore":
            std = self._get_stats_view(x, self.std)
            return (x * std) + mean
            
        if self.mode == "symrange":
            denom = self._get_stats_view(x, self.denom)
            return (x * denom / self.norm_const) + mean
            
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
        window_size: int,
        variable_indices: Sequence[int],
        dtype: Literal["float32", "float16"],
        data_reshape: Tuple[int,int] | None, 
        normalizer: Normalizer,
    ) -> None:
        super().__init__()
        self.indices = indices 
        self.window_size = window_size
        assert self.window_size >= 3, f"Window size must be >= 3, got {self.window_size}"

        self.var_idx = list(variable_indices)
        
        self.dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        self.data_reshape = data_reshape
        self.normalizer = normalizer
        self.out_dtype = self.dtype_map[dtype]
        
        # Lazy open with xarray
        self.ds = xr.open_dataset(dataset_path, engine='zarr', zarr_format=2 )
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
        if self.data_reshape is not None:
            data_block = data_block.reshape(*data_block.shape[:-1],*self.data_reshape)

        # 3. Create Tensors directly (No CPU Normalization loop)
        # We split Boundary (x) and Internal (y)
        # x: First and Last frame
        # y: Intermediate frames
        

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

        return self.normalizer.forward(x_t) , self.normalizer.forward(y_t), meta

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
        data_reshape: Optional[Tuple[int,int]] | None = None,
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
        self.data_reshape = data_reshape
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
        ds = xr.open_dataset(self.dataset_path, engine='zarr', zarr_format=2 )
        total_time = ds.sizes['time']
        
        # 1. Resolve Variable Names to Indices
        all_var_names = ds.attrs["variables"]
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
                mean = ds['mean'].values 
                std = ds['stdev'].values
                minimum= ds['minimum'].values
                maximum= ds['maximum'].values
                self.normalizer = Normalizer(
                    mode=self.normalize_mode,
                    mean=mean[self.var_indices],
                    std=std[self.var_indices],
                    minimum=minimum[self.var_indices],
                    maximum=maximum[self.var_indices],
                    norm_const=self.norm_const,
                )
                                
            except KeyError as e:
                rank_zero_info(f"Warning: Could not load stat {e} from Zarr. Normalization might fail.")
                self.normalizer = Normalizer(mode="none")


      
        # 3. Split Indices
        train_idx, val_idx, test_idx = self._get_split_indices(total_time)
        
        rank_zero_info(f"Total valid windows: {len(train_idx)+len(val_idx)+len(test_idx)}")
        rank_zero_info(f"Train samples: {len(train_idx)}")
        rank_zero_info(f"Val samples:   {len(val_idx)}")
        rank_zero_info(f"Test samples:  {len(test_idx)}")

        if self.data_reshape is not None:
            rank_zero_info(f"Data will be reshaped to: {self.data_reshape}")
          
        # 4. Create Datasets
        self.train_ds = MEPSZarrDataset(
            self.dataset_path, train_idx, self.window_size, self.var_indices, self.dtype
            ,self.data_reshape, self.normalizer
        )
        self.val_ds = MEPSZarrDataset(
            self.dataset_path, val_idx, self.window_size, self.var_indices, self.dtype
            ,self.data_reshape, self.normalizer
        )
        self.test_ds = MEPSZarrDataset(
            self.dataset_path, test_idx, self.window_size, self.var_indices, self.dtype
            ,self.data_reshape, self.normalizer
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

    #TODO: to be used later
    # def get_normalizer(self) -> nn.Module:
    #     if hasattr(self, "normalizer"):
    #         return self.normalizer
    #     raise RuntimeError(
    #         "Normalizer is not initialized. "
    #         "Call setup() before accessing the normalizer."
    #     )