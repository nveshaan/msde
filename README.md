# Mean-Shift Density Enhancement

MSDE built from four swappable components.

## Setup

```bash
git clone https://github.com/nveshaan/msde.git
cd msde
uv sync
```

## Layout

```
msde/
  module.py        MeanShiftDensityEnhancement: composes the four components
  neighbors/       NeighborSelector -> EuclideanKNN, CosineKNN, WhitenedKNN, JointKNN
  weights/         DensityWeights   -> SparseEmpirical, KDEWeights, GravityWeights, UniformWeights
  step/            MovementStep     -> PlainBarycenter, GravityBarycenter
  loop/            ShiftLoop        -> EarlyStopLoop
  common/          code shared by more than one component
tests/             regression tests against the pre-refactor outputs
```

See `DOC.md` for the API and how to add a new strategy. Run the tests with `uv run pytest`.

## Acknowledgements

- Kar, P., Bordoloi, R., Wolkenhauer, O., & Bej, S. (2026). Anomaly Detection via Mean Shift Density Enhancement. arXiv:2602.03293. https://doi.org/10.48550/arXiv.2602.03293

- Kar, P., Lakshmi, G., & Bej, S. (2026). Improved Anomaly Detection in Medical Images via Mean Shift Density Enhancement. arXiv:2604.19191. https://doi.org/10.48550/arXiv.2604.19191

## License

This project is distributed under the MIT License. See the `LICENSE` file for details.
