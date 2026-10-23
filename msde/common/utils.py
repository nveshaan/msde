import logging

import torch

logger = logging.getLogger("msde")

DEFAULT_DEVICE = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"

# Constants shared by every differentiable (soft) strategy.
DIFF_EPS = 1e-8
DIFF_LOG_SCALE_EPS = 1e-6
DIFF_SOLVER_ITERATIONS = 25
DEFAULT_DIFF_TEMPERATURE = 0.1


def prepare_scalar(value, name, device, dtype, learnable):
    if value is None:
        return None

    scalar = torch.as_tensor(value, device=device, dtype=dtype)
    if scalar.numel() != 1:
        raise ValueError(f"{name} must be a scalar")

    if learnable:
        if not scalar.is_leaf or not scalar.requires_grad:
            scalar = scalar.detach().clone().requires_grad_(True)
    else:
        scalar = scalar.detach()
    return scalar



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


class ParametrizedModule(torch.nn.Module):
    """Base for the four MSDE components: registers scalar hyperparameters.

    A scalar is either a buffer (fixed) or a ``torch.nn.Parameter`` (learnable);
    learnable ones are listed in ``self.learnable_parameters``. Passing an
    existing ``nn.Parameter`` as ``value`` registers that very object, so a
    single learnable scalar (e.g. a temperature) can be shared by components.
    """

    def __init__(self):
        super().__init__()
        self.learnable_parameters = {}

    def register_scalar(self, name, value, learnable=False, device=None, dtype=None):
        if isinstance(value, torch.nn.Parameter):
            setattr(self, name, value)
            if learnable:
                self.learnable_parameters[name] = value
            return

        if dtype is None:
            dtype = value.dtype if isinstance(value, torch.Tensor) and value.is_floating_point() \
                else torch.get_default_dtype()
        scalar = prepare_scalar(value, name, device, dtype, learnable and value is not None)
        if scalar is not None and learnable:
            parameter = torch.nn.Parameter(scalar)
            setattr(self, name, parameter)
            self.learnable_parameters[name] = parameter
        else:
            self.register_buffer(name, scalar)

    def replace_scalar_with_parameter(self, name, value):
        """Swap a (None) buffer for a learnable Parameter after construction."""
        self._buffers.pop(name, None)
        parameter = torch.nn.Parameter(value)
        setattr(self, name, parameter)
        self.learnable_parameters[name] = parameter
        return parameter
