# Contagious disruptions and complexity traps in economic development

[![DOI](https://zenodo.org/badge/23976/cbrummitt/Contagious_disruptions_complexity_trap_economic_development.svg)](https://zenodo.org/badge/latestdoi/23976/cbrummitt/Contagious_disruptions_complexity_trap_economic_development)

This repository contains the code used to produce the figures and results in the paper

> Charles D. Brummitt , Kenan Huremovic, Paolo Pin, Matthew H. Bonds, and Fernando Vega-Redondo. "Contagious disruptions and complexity traps in economic development". Working paper, 2017.

Below is an explanation of the contents of this repository and of the depencies of the code.

## Scripts and notebooks

The folder [`scripts`](scripts) contains code and notebooks for replicating the results of the paper. Instructions for how to replicate each figure are given in the notebooks below.

### Mathematica notebooks and Wolfram Language Code

* [`figures_1_6.nb`](scripts/figures_1_6.nb) combines several datasets, runs some simulations of the model (for figure 6b), and creates figures 1 and 6;
* [`figures_3_4.nb`](scripts/figures_3_4.nb) creates the phase portraits in figures 3 and 4;
* [`figure5.nb`](scripts/figure5.nb) creates the phase diagram in figure 5;
* [`figure2.key`](scripts/figure2.key) is a Keynote file containing figure 2;
* [`set_strategies_that_could_be_best_response.wl`](scripts/set_strategies_that_could_be_best_response.wl) is a script in the Wolfram Language that computes the set of strategies that could be a best response and the chance of successfully producing. The notebook [`Explore the set of strategies that could be a best response.nb`](Explore the set of strategies that could be a best response.nb) contains a few examples of the set of strategies that could be a best response.

**Dependencies**: The _Mathematica_ notebooks were made in version 11.0. They will work in version 10.4 as well. These notebooks can be viewed for free using the [Wolfram CDF player](https://www.wolfram.com/cdf-player/), available on Windows, Mac, and Linux, and they can be run for free in the [Wolfram Cloud](http://develop.wolframcloud.com/).

### Jupyter notebook and Python code 

* The Jupyter notebook [`Make_Figure_7.ipynb`](scripts/Make_Figure_7.ipynb) contains the Python code used to create Figure 7. This notebook uses the following code written in Python for simulating the model:
	* [`ABM.py`](scripts/ABM.py) defines an economy and agents;
	* [`EconomySimulator.py`](scripts/EconomySimulator.py) (which is a wrapper for `ABM.Economy` that simulates the economy and collects information about it).

**Dependencies**: The Python code was run with [Python](https://www.python.org/) `3.5.2`, [Matplotlib](http://matplotlib.org/) `2.0.0`, [Pandas](http://pandas.pydata.org/) `0.19.2`, [Seaborn](http://seaborn.pydata.org/) `0.7.1`, [NumPy](http://www.numpy.org/) `1.11.3`, [progressbar](https://pypi.python.org/pypi/progressbar2#downloads) `3.12.0`, and [joblib](https://pypi.python.org/pypi/joblib) `0.10.3`. The first five of these come with the [installation of Anaconda](https://www.continuum.io/downloads); the latter two can be installed using [`pip`](https://pypi.python.org/pypi/pip) by running `pip install progressbar` and `pip install joblib` from a terminal.

## Figures

The folder [`figures`](figures) contains the seven figures as PDF files. It also contains the figure [`compare_std_dev_F.pdf`](figures/compare_std_dev_F.pdf) made at the end of the Jupyter notebook [`Make_Figure_7.ipynb`](scripts/Make_Figure_7.ipynb), which illustrates the difference in standard deviations of 200 simulated time-series like those shown in Figure 7(b).

## Empirical data

Empirical datasets are located in the folder [`empirical_data`](empirical_data). This data is described, cleaned, and used in the notebook [`figures_1_6.nb`](scripts/figures_1_6.nb), located in the [`scripts`](scripts) folder. 

## Simulated data

Data generated from simulating the model was exported into the folder [`scripts/simulated_data`](scripts/simulated_data) by the notebooks [`figures_3_4.nb`](scripts/figures_3_4.nb), [`figures_5.nb`](scripts/figures_5.nb), [`figures_1_6.nb`](scripts/figures_1_6.nb), and [`Make_Figure_7.ipynb`](scripts/Make_Figure_7.ipynb) (all these notebooks are located in the [`scripts`](scripts) folder). 