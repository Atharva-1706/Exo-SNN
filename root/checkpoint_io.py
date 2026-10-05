"""Safe, version-tolerant checkpoint loading shared by inference and training."""
import warnings

import torch


def load_checkpoint(path):
    """Load a checkpoint onto the CPU.

    Prefers ``weights_only=True`` (no arbitrary pickle execution). Checkpoints
    that carry extra non-tensor metadata (e.g. numpy scalars/arrays in the
    saved validation metrics) are retried with ``weights_only=False`` and a
    warning, so a newer PyTorch default never makes a valid project checkpoint
    unreadable. Only load checkpoints you produced or trust.
    """
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except Exception as first_error:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                ckpt = torch.load(path, map_location="cpu", weights_only=False)
        except Exception as second_error:
            raise RuntimeError(
                f"Could not read checkpoint '{path}'. The file may be corrupt or truncated "
                f"({type(second_error).__name__}: {second_error})."
            ) from second_error
        print(f"   NOTE: '{path}' contains non-tensor metadata; loaded with weights_only=False "
              f"({type(first_error).__name__}). Only load checkpoints you trust.")
        return ckpt
