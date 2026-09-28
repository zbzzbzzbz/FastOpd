"""Opt-in Student CUDA memory snapshots for the OPD loop."""

import json
import subprocess
import time

import torch

_ACTIVE_PROBE = None


def set_active_probe(probe):
    global _ACTIVE_PROBE
    _ACTIVE_PROBE = probe


def detail(event, **extra):
    probe = _ACTIVE_PROBE
    if probe is not None and probe.current_stage:
        return probe.snapshot(f"{probe.current_stage}:{event}", **extra)
    return None


def tensor_bytes(tensors):
    seen = set()
    total = 0
    for tensor in tensors:
        if tensor is None or not isinstance(tensor, torch.Tensor):
            continue
        key = id(tensor)
        if key not in seen:
            seen.add(key)
            total += tensor.numel() * tensor.element_size()
    return total


def optimizer_tensors(optimizer):
    for state in optimizer.state.values():
        for value in state.values():
            if isinstance(value, torch.Tensor):
                yield value


def offload_optimizer_state(optimizer):
    moved_bytes = 0
    for state in optimizer.state.values():
        for key, value in state.items():
            if isinstance(value, torch.Tensor) and value.is_cuda:
                moved_bytes += value.numel() * value.element_size()
                state[key] = value.detach().cpu()
    return moved_bytes


class StudentMemoryProbe:
    def __init__(self, path, device):
        self.path = path
        self.device = device
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            raise FileExistsError(self.path)
        self.open_scopes = {}
        self.current_stage = None
        self.global_peak_allocated = 0
        self.global_peak_reserved = 0

    def snapshot(self, event, *, model=None, optimizer=None, **extra):
        torch.cuda.synchronize(self.device)
        timestamp = time.time()
        used = int(
            subprocess.check_output(
                [
                    "nvidia-smi",
                    "-i",
                    str(self.device.index or 0),
                    "--query-gpu=memory.used",
                    "--format=csv,noheader,nounits",
                ],
                text=True,
            ).strip()
        )
        row = {
            "time": timestamp,
            "event": event,
            "gpu_used_mib": used,
            "allocated_bytes": torch.cuda.memory_allocated(self.device),
            "reserved_bytes": torch.cuda.memory_reserved(self.device),
            "max_allocated_bytes": torch.cuda.max_memory_allocated(self.device),
            "max_reserved_bytes": torch.cuda.max_memory_reserved(self.device),
            **extra,
        }
        self.global_peak_allocated = max(
            self.global_peak_allocated, row["max_allocated_bytes"]
        )
        self.global_peak_reserved = max(
            self.global_peak_reserved, row["max_reserved_bytes"]
        )
        if model is not None:
            parameters = list(model.parameters())
            row.update(
                {
                    "parameter_bytes": tensor_bytes(parameters),
                    "trainable_parameter_bytes": tensor_bytes(
                        p for p in parameters if p.requires_grad
                    ),
                    "gradient_bytes": tensor_bytes(p.grad for p in parameters),
                    "buffer_bytes": tensor_bytes(model.buffers()),
                    "parameter_count": sum(p.numel() for p in parameters),
                    "trainable_parameter_count": sum(
                        p.numel() for p in parameters if p.requires_grad
                    ),
                }
            )
        if optimizer is not None:
            state = list(optimizer_tensors(optimizer))
            row.update(
                {
                    "optimizer_state_bytes": tensor_bytes(state),
                    "optimizer_state_gpu_bytes": tensor_bytes(
                        x for x in state if x.is_cuda
                    ),
                    "optimizer_state_cpu_bytes": tensor_bytes(
                        x for x in state if not x.is_cuda
                    ),
                    "optimizer_state_tensor_count": len(state),
                }
            )
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row) + "\n")
        return row

    def begin(self, name, *, model=None, optimizer=None, **extra):
        if name in self.open_scopes:
            raise RuntimeError(f"scope already open: {name}")
        torch.cuda.synchronize(self.device)
        torch.cuda.reset_peak_memory_stats(self.device)
        row = self.snapshot(f"{name}:before", model=model, optimizer=optimizer, **extra)
        self.open_scopes[name] = row["time"]
        return row

    def end(self, name, *, model=None, optimizer=None, **extra):
        if name not in self.open_scopes:
            raise RuntimeError(f"scope not open: {name}")
        started = self.open_scopes.pop(name)
        return self.snapshot(
            f"{name}:after",
            model=model,
            optimizer=optimizer,
            elapsed_seconds=time.time() - started,
            **extra,
        )
