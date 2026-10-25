"""Build a model from a YAML experiment config.

    model = build_model(load_config("configs/default.yaml", ["neighbors.k=10"]))

Schema (every key optional; an omitted component gets its default):

    vars: {...}              free-form values for ${vars.x} interpolation
    device: auto             "auto", "cpu", "cuda" or "mps"; passed to every component
    label_mode: same_label   or zero_label
    log_file: null
    neighbors: {name: EuclideanKNN, k: 20, ...}
    weights:   {name: GravityWeights, density: {name: KDEWeights}, ...}
    step:      {name: GravityBarycenter, ...}
    loop:      {name: EarlyStopLoop, ...}

``name`` is a class name from ``msde`` or a dotted import path to your own
strategy (``my_pkg.my_module.MyStep``); the remaining keys are its constructor
options. A component option whose value is a mapping with a ``name`` is built
as a component itself (e.g. ``GravityWeights.density``).

Run ``python -m msde.config configs/default.yaml [key=value ...]`` to check a
config: it prints the resolved config and the model it builds.
"""
import importlib
import inspect
import sys

from omegaconf import OmegaConf

import msde

from .common.utils import DEFAULT_DEVICE
from .loop import ShiftLoop
from .neighbors import ChunkedKNN, NeighborSelector
from .step import MovementStep
from .weights import DensityWeights

SECTIONS = {
    "neighbors": NeighborSelector,
    "weights": DensityWeights,
    "step": MovementStep,
    "loop": ShiftLoop,
}
TOP_LEVEL_KEYS = ("vars", "device", "label_mode", "log_file", *SECTIONS)
_ABSTRACT = {NeighborSelector, ChunkedKNN, DensityWeights, MovementStep, ShiftLoop}


def available_components():
    """``{section: sorted class names}`` of the strategies shipped with msde."""
    return {
        section: sorted(
            name for name, cls in vars(msde).items()
            if isinstance(cls, type) and issubclass(cls, base) and cls not in _ABSTRACT
        )
        for section, base in SECTIONS.items()
    }


def load_config(path, overrides=()):
    """Load a YAML config, apply ``key=value`` overrides and resolve interpolations.

    Returns a plain ``dict``.
    """
    cfg = OmegaConf.load(path)
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(list(overrides)))
    return OmegaConf.to_container(cfg, resolve=True)


def _resolve_class(name, bases, where):
    if not isinstance(name, str):
        raise ValueError(f"{where}: 'name' must be a string; got {name!r}")
    if "." in name:
        module_name, _, class_name = name.rpartition(".")
        try:
            cls = getattr(importlib.import_module(module_name), class_name)
        except (ImportError, AttributeError) as e:
            raise ValueError(f"{where}: cannot import {name!r} ({e})") from e
    else:
        cls = getattr(msde, name, None)
    if not (isinstance(cls, type) and issubclass(cls, bases)) or cls in _ABSTRACT:
        known = sorted(n for n, c in vars(msde).items()
                       if isinstance(c, type) and issubclass(c, bases) and c not in _ABSTRACT)
        raise ValueError(f"{where}: {name!r} is not a usable {'/'.join(b.__name__ for b in bases)}; "
                         f"available: {known}")
    return cls


def _build_component(spec, bases, device, where):
    if not isinstance(spec, dict) or "name" not in spec:
        raise ValueError(f"{where}: expected a mapping with a 'name' key; got {spec!r}")
    options = dict(spec)
    cls = _resolve_class(options.pop("name"), bases, where)

    params = inspect.signature(cls.__init__).parameters
    valid = [n for n, p in params.items() if n != "self"
             and p.kind not in (p.VAR_POSITIONAL, p.VAR_KEYWORD)]
    unknown = sorted(set(options) - set(valid))
    if unknown:
        raise ValueError(f"{where}: {cls.__name__} has no option(s) {unknown}; valid options: {valid}")

    for key, value in options.items():                       # nested components, e.g. GravityWeights.density
        if isinstance(value, dict) and "name" in value:
            options[key] = _build_component(value, tuple(SECTIONS.values()), device, f"{where}.{key}")
    if "device" in params and "device" not in options:
        options["device"] = device
    return cls(**options)


def build_model(cfg):
    """Build a ``MeanShiftDensityEnhancement`` from a config mapping (see module docstring)."""
    if not isinstance(cfg, dict):
        raise ValueError(f"config must be a mapping; got {type(cfg).__name__}")
    unknown = sorted(set(cfg) - set(TOP_LEVEL_KEYS))
    if unknown:
        raise ValueError(f"unknown top-level key(s) {unknown}; valid keys: {list(TOP_LEVEL_KEYS)}")

    device = cfg.get("device", "auto")
    if device in (None, "auto"):
        device = DEFAULT_DEVICE

    components = {
        section: _build_component(cfg[section], (base,), device, section)
        for section, base in SECTIONS.items() if cfg.get(section) is not None
    }
    return msde.MeanShiftDensityEnhancement(
        **components,
        label_mode=cfg.get("label_mode", "same_label"),
        device=device,
        log_file=cfg.get("log_file"),
    )


def build_from_file(path, overrides=()):
    """``build_model(load_config(path, overrides))``."""
    return build_model(load_config(path, overrides))


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if not argv or argv[0] in ("-h", "--help"):
        print(__doc__)
        print("available components:")
        for section, names in available_components().items():
            print(f"  {section}: {', '.join(names)}")
        return 0
    cfg = load_config(argv[0], argv[1:])
    print(OmegaConf.to_yaml(OmegaConf.create(cfg)))
    print(build_model(cfg))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
