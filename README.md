# Contagious disruptions and complexity traps in economic development

[![DOI](https://zenodo.org/badge/23976/cbrummitt/Contagious_disruptions_complexity_trap_economic_development.svg)](https://zenodo.org/badge/latestdoi/23976/cbrummitt/Contagious_disruptions_complexity_trap_economic_development)

This repository contains the code used to produce the figures and results in the paper

> Charles D. Brummitt , Kenan Huremovic, Paolo Pin, Matthew H. Bonds, and Fernando Vega-Redondo. "Contagious disruptions and complexity traps in economic development". Working paper, 2016.

## Contents

### Scripts and notebooks

* [`figures_1_6.nb`](figures_1_6.nb) combines several datasets, runs some simulations of the model (for figure 6b), and creates figures 1 and 6
* [`figures_3_4.nb`](figures_3_4.nb) creates the phase portraits in figures 3 and 4
* [`figure5.nb`](figure5.nb) creates the phase diagram in figure 5
* [`figure2.key`](figure2.key) is a Keynote file containing figure 2
* [`set_strategies_that_could_be_best_response.wl`](set_strategies_that_could_be_best_response.wl) is a script in the Wolfram Language that computes the set of strategies that could be a best response and the chance of successfully producing. The notebook [`Explore the set of strategies that could be a best response.nb`](Explore the set of strategies that could be a best response.nb) contains a few examples of the set of strategies that could be a best response.

**Dependencies**: The _Mathematica_ notebooks were made in version 11.0. They will work in version 10.4 as well.

### Figures

The folder [figures](figures) contains the six figures as PDF files.

### Empirical data

Empirical datasets are located in the folder [empirical_data](empirical_data). This data is described, cleaned and used in the notebook [figures_1_6.nb](figures_1_6.nb). 

### Simulated data

Simulated data was exported into the folder [simulated_data](simulated_data) by the notebooks [figures_3_4.nb](figures_3_4.nb), [figures_5.nb](figures_5.nb), and [figures_1_6.nb](figures_1_6.nb). 