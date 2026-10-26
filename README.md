# Mean-Shift Density Enhancement

Mean-shift density enhancement of unit-normalised embeddings, in a single file: `src/msde.py`.

## Setup

```bash
git clone https://github.com/nveshaan/msde.git
cd msde
uv sync
```

```python
from msde import MeanShiftDensityEnhancement

shifted, movement, trajectory = MeanShiftDensityEnhancement(k=50)(X)
```

See `DOC.md` for the pipeline, every option, how to replace the selection score and the neighbour weights, and speed notes.
A modular version, with swappable neighbour, weight, step and loop components and YAML experiment configs, lives on the `modular` branch.

## Acknowledgements

- Kar, P., Bordoloi, R., Wolkenhauer, O., & Bej, S. (2026). Anomaly Detection via Mean Shift Density Enhancement. arXiv:2602.03293. https://doi.org/10.48550/arXiv.2602.03293

- Kar, P., Lakshmi, G., & Bej, S. (2026). Improved Anomaly Detection in Medical Images via Mean Shift Density Enhancement. arXiv:2604.19191. https://doi.org/10.48550/arXiv.2604.19191

## License

This project is distributed under the MIT License. See the `LICENSE` file for details.
