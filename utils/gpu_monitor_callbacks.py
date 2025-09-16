# utils/gpu_monitors.py
from __future__ import annotations
import os, torch
import lightning as L

class GPUsAssignedCallback(L.Callback):
    def on_fit_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        r = trainer.global_rank
        lr = getattr(trainer, "local_rank", None)
        dev_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
        cur = torch.cuda.current_device() if torch.cuda.is_available() else -1
        print(
            f"[Start] rank={r} local_rank={lr} "
            f"cuda_count={dev_count} current_device={cur} "
            f"CVD={os.environ.get('CUDA_VISIBLE_DEVICES')}",
            flush=True,
        )

class GPUMemoryMonitorCallback(L.Callback):
    def __init__(self, every_n_steps: int = 50) -> None:
        self.every_n_steps = int(every_n_steps)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx) -> None:  # noqa: ARG002
        if (trainer.global_step + 1) % self.every_n_steps != 0:
            return
        if not torch.cuda.is_available():
            return
        idx = torch.cuda.current_device()
        alloc = torch.cuda.memory_allocated(idx) / (1024**2)
        reserv = torch.cuda.memory_reserved(idx) / (1024**2)
        print(
            f"GPUMemoryMonitor: [Rank {trainer.global_rank}] "
            f"GPU {idx}: allocated={alloc:.1f} MB, reserved={reserv:.1f} MB",
            flush=True,
        )
