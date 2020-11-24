<div align="center">
  <img src= "https://github.com/vd1371/PyImbalReg/blob/main/xtra/banner.png">
</div>

# PyImbalReg
## Pre-processing technics for imbalanced datasets in regression modelling

[![PyPI version](https://badge.fury.io/py/PyImbalReg.svg)](https://badge.fury.io/py/PyImbalReg)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Codacy Badge](https://api.codacy.com/project/badge/Grade/525f3e4f9261425eba1e40ff2b1d7710)](https://app.codacy.com/gh/vd1371/PyImbalReg?utm_source=github.com&utm_medium=referral&utm_content=vd1371/PyImbalReg&utm_campaign=Badge_Grade)
![GitHub last commit](https://img.shields.io/github/last-commit/vd1371/PyImbalReg)

---
### Dealing with imbalanced datasets for regression modelling
Your trained regression model has heteroskedasticity problem?
Your model can't predict the extreme values very well?
Consider using these pre-processing technics for solving these issues probably caused by your imbalanced dataset.

  - RandomOversampling (RO)
  - GaussianNoise and Undersampling (GN)
  - WEighted Relevance-based Combination Strategy (WERCS)
---
## How to use? (2 Minutes read)
 1.  Pass your data as a pandas dataframe to any of these technics
 2.  Define a releavnce function that maps the output variable to [0, 1] (The higher this value the rarer the samples)
 3.  Set a threshold for flagging the samples: rare and normal
 4.  Set model-specific parameters
 5.  get() a new dataset with new samples

---
## Installation
```python
## Pypi version
pip install PyImbalReg

## GitHub version
pip install git+https://github.com/vd1371/PyImbalReg.git
```
---
## Example

```python
# importing PyImbalReg
import PyImbalReg as pir

# importing the data
from seaborn import load_dataset
data = load_dataset('dots')

ro = pir.RandomOversampling(data,           # Passing the data
			rel_func = None,    # Default relevance function will be used
			threshold = 0.7,    # Set the threshold
			o_percentage = 5    # ( o_percentage - 1 ) x n_rare_samples will be added 
			)
new_data = ro.get()
```
---
## Requirements
 1. SciPy
 2. Pandas
 3. Numpy

---
## Other examples

 1. [RandomOversampling](https://github.com/vd1371/PyImbalReg/blob/main/tests/Example-RO.py)
 2. [GaussianNoise](https://github.com/vd1371/PyImbalReg/blob/main/tests/Example-GN.py)
 3. [WERCS](https://github.com/vd1371/PyImbalReg/blob/main/tests/Example-WERCS.py)
---
## Contributions

Please share your issues, new technics and your contributions with us.
Your help is much appreciated in advance.

---
## If you are using this repository

### Please cite the below reference(s)

Branco, P., Torgo, L. and Ribeiro, R.P., 2019.
Pre-processing approaches for imbalanced distributions in regression.
Neurocomputing, 343, pp.76-99.

---
## License
© Vahid Asghari, 2020. Licensed under the General Public License v3.0 (GPLv3).

P.S. Some parts of the readme and codes were inspired from https://github.com/nickkunz/smogn

---