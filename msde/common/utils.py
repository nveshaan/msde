import logging

import torch

logger = logging.getLogger("msde")


def _raise_dynamo_recompile_limit(limit=128):
    """Every compiled kernel built from the same factory shares one code object,
    and dynamo counts recompiles per code object (default limit 8). Trying a
    handful of k values, dtypes or shapes would otherwise hit the limit, which
    is a hard error under ``fullgraph=True``."""
    try:
        import torch._dynamo.config as dynamo_config
        for name in ("recompile_limit", "cache_size_limit"):
            if hasattr(dynamo_config, name):
                setattr(dynamo_config, name, max(getattr(dynamo_config, name), limit))
    except Exception:       # a torch build without dynamo: nothing to raise
        pass


_raise_dynamo_recompile_limit()

DEFAULT_DEVICE = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"


def prepare_scalar(value, name, device, dtype):
    if value is None:
        return None

    scalar = torch.as_tensor(value, device=device, dtype=dtype)
    if scalar.numel() != 1:
        raise ValueError(f"{name} must be a scalar")
    return scalar.detach()


def configure_logging(log_file):
    """Configure this module's file-only logger."""
    for h in list(logger.handlers):
        logger.removeHandler(h)
        h.close()

    logger.propagate = False

    if log_file:
        handler = logging.FileHandler(log_file)
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(name)s %(levelname)s %(message)s")
        )
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)
    else:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s %(name)s %(levelname)s %(message)s"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)



_SPARSE_MM_SUPPORT_CACHE = {}

def sparse_mm_supported(device):
    """Return whether sparse matrix multiplication works on ``device``."""
    if device in _SPARSE_MM_SUPPORT_CACHE:
        return _SPARSE_MM_SUPPORT_CACHE[device]
    try:
        idx = torch.zeros((2, 1), dtype=torch.long, device=device)
        vals = torch.ones(1, device=device)
        t = torch.sparse_coo_tensor(idx, vals, (1, 1), device=device).coalesce()
        torch.sparse.mm(t, torch.ones(1, 1, device=device))
        supported = True
    except Exception:
        supported = False
    _SPARSE_MM_SUPPORT_CACHE[device] = supported
    return supported


class ScalarModule(torch.nn.Module):
    """Base for the four MSDE components: registers scalar hyperparameters as buffers."""

    def register_scalar(self, name, value, device=None, dtype=None):
        if dtype is None:
            dtype = value.dtype if isinstance(value, torch.Tensor) and value.is_floating_point() \
                else torch.get_default_dtype()
        self.register_buffer(name, prepare_scalar(value, name, device, dtype))
