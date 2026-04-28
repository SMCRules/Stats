# https://www.pymc.io/projects/examples/en/latest/generalized_linear_models/multilevel_modeling.html#id32
# https://bambinos.github.io/bambi/notebooks/radon_example.html

import os
import warnings

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt
import seaborn as sns
import xarray as xr
print(f"Running on PyMC v{pm.__version__}")

# Get radon data
path = "https://raw.githubusercontent.com/pymc-devs/pymc-examples/main/examples/data/srrs2.dat"
radon_df = pd.read_csv(path)

# Get city data
city_df = pd.read_csv(pm.get_data("cty.dat"))

display(radon_df.head())
print(radon_df.shape[0])
