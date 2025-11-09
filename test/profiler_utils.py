import os
from datetime import datetime
from typing import Optional

import torch
from torch.profiler import ProfilerActivity
from torch.profiler import profile
from torch.profiler import schedule
from torch.profiler import tensorboard_trace_handler
from torch.utils.tensorboard import SummaryWriter


def _ensure_event_file(log_dir):
    """
    TensorBoard expects at least one TB event file in the run directory before it
    enumerates available profile traces. Touch one via SummaryWriter (and close it)
    so we stay compatible without adding extra logging overhead.
    """
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir)
    writer.close()


def _profile_run_dir(log_dir: str, run_name: Optional[str] = None) -> str:
    """
    TensorBoard's profile plugin expects traces under plugins/profile/<run_name>/.
    """
    if run_name is None:
        run_name = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = os.path.join(log_dir, "plugins", "profile", run_name)
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


class StepProfiler:
    def __init__(
        self,
        log_dir="out/prof_qwen",
        *,
        wait=1,
        warmup=1,
        active=3,
        repeat=1,
        run_name=None,
    ):
        _ensure_event_file(log_dir)
        profile_dir = _profile_run_dir(log_dir, run_name)
        activities = [ProfilerActivity.CPU]
        if torch.cuda.is_available():
            activities.append(ProfilerActivity.CUDA)
        self.prof = profile(
            activities=activities,
            schedule=schedule(wait=wait, warmup=warmup, active=active, repeat=repeat),
            on_trace_ready=tensorboard_trace_handler(profile_dir),
            profile_memory=True,
            record_shapes=True,
            with_stack=True,
        )

    def __enter__(self):
        self.prof.__enter__()
        return self

    def __exit__(self, et, ex, tb):
        self.prof.__exit__(et, ex, tb)

    def step(self):
        self.prof.step()
