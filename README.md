<div align="center">
  <img src="https://github.com/vd1371/PyImbalReg/blob/main/xtra/banner.png" alt="PyImbalReg">
</div>

# PyImbalReg

## Pre-processing techniques for imbalanced datasets in regression

[![PyPI version](https://badge.fury.io/py/PyImbalReg.svg)](https://badge.fury.io/py/PyImbalReg)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Codacy Badge](https://api.codacy.com/project/badge/Grade/525f3e4f9261425eba1e40ff2b1d7710)](https://app.codacy.com/gh/vd1371/PyImbalReg?utm_source=github.com&utm_medium=referral&utm_content=vd1371/PyImbalReg&utm_campaign=Badge_Grade)
![GitHub last commit](https://img.shields.io/github/last-commit/vd1371/PyImbalReg)

---

### Dealing with imbalanced datasets for regression

Your trained regression model has a heteroskedasticity problem? It can't predict extreme values well? These pre-processing techniques can help when the issues are caused by an imbalanced target distribution.

- **Random Oversampling (RO)**
- **Gaussian Noise and Undersampling (GN)**
- **Weighted Relevance-based Combination Strategy (WERCS)**

---

### How to use (2-minute read)

1. Pass your data as a pandas DataFrame to any of the techniques.
2. Define a **relevance function** that maps the target variable to [0, 1] (higher value = rarer samples).
3. Set a **threshold** to flag rare vs normal samples.
4. Set method-specific parameters (e.g. oversampling/undersampling ratios).
5. Call `.get()` to obtain the resampled dataset.

---

## Installation

### From PyPI (pip)

```bash
pip install PyImbalReg
```

### From PyPI with uv

```bash
uv pip install PyImbalReg
```

### From GitHub

```bash
pip install git+https://github.com/vd1371/PyImbalReg.git
```

### Development (uv)

This project uses [uv](https://docs.astral.sh/uv/) for fast, reliable dependency management.

1. **Install uv** (if needed):

   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   # or: pip install uv
   ```

2. **Clone and sync**:

   ```bash
   git clone https://github.com/vd1371/PyImbalReg.git
   cd PyImbalReg
   uv sync
   ```

   This creates a virtual environment, installs the package in editable mode, and installs dev dependencies (e.g. pytest, seaborn).

3. **Run tests** (with uv):

   ```bash
   uv run pytest tests/
   ```

4. **Lock dependencies** (optional):

   ```bash
   uv lock
   ```

   Commit `uv.lock` for reproducible installs.

---

## Example

```python
import PyImbalReg as pir
from seaborn import load_dataset

data = load_dataset("dots")

ro = pir.RandomOversampling(
    df=data,
    rel_func="default",
    threshold=0.7,
    o_percentage=5,  # (o_percentage - 1) × n_rare_samples will be added
)
new_data = ro.get()
```

---

## Requirements

- Python ≥ 3.8
- NumPy
- Pandas
- SciPy

(All are declared in `pyproject.toml` and installed automatically with the package.)

---

## More examples

- [Random Oversampling](https://github.com/vd1371/PyImbalReg/blob/main/tests/Example-RO.py)
- [Gaussian Noise](https://github.com/vd1371/PyImbalReg/blob/main/tests/Example-GN.py)
- [WERCS](https://github.com/vd1371/PyImbalReg/blob/main/tests/Example-WERCS.py)

---

## Contributing

Issues, new techniques, and pull requests are welcome.

---

## Citation

If you use this repository, please cite:

> Branco, P., Torgo, L. and Ribeiro, R.P., 2019.  
> Pre-processing approaches for imbalanced distributions in regression.  
> *Neurocomputing*, 343, pp.76–99.

---

## License

© Vahid Asghari, 2020. Licensed under the GNU General Public License v3.0 (GPLv3).

*Some parts of the README and code were inspired by [smogn](https://github.com/nickkunz/smogn).*
