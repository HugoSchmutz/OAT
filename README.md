# OAT: Online Active Testing

Code for **Online Active Testing: Adaptive Importance Sampling for Unbiased Risk Estimation in Data Streams**.

Links: [OpenReview](https://openreview.net/forum?id=AyJOqDpoNN#discussion) | [Paper](...) | [amU.HAL.science]()

## Overview

Evaluating a model requires labelled test data, and labels are often expensive. *Active testing* reduces this cost by choosing which test points to label, instead of labelling them uniformly at random. Selecting points adaptively introduces a sampling bias, which importance weighting can correct.

**OAT** extends this idea to the online (streaming) setting: data points arrive one at a time, and the decision to request a label must be made as each point arrives. The goal is an unbiased estimate of the model's risk that uses fewer labels than uniform sampling to reach lower variance, i.e. better risk estimation accuracy.


## Repository structure

```
OAT/
├── OAT/                                  # Core implementation of the OAT method: OAT estimator, 1st and 2nd moment 
|                                         # sampling for various loss function
├── models/                               # Surrogate GP with computation of third and fourth moments 
├── utils/                                # Helper functions
├── utils_visu.py                         # Plotting and visualisation utilities
├── SyntheticData_Classification.ipynb    # Experiments on synthetic classification data
└── SyntheticData_Regression.ipynb        # Experiments on synthetic regression data
```


## Mainly two notebooks:

The quickest way to see OAT in action is through the notebooks that can be used to reproduce Figure 2.

- **`SyntheticData_Classification.ipynb`**: OAT of a Random Forest on the two moons dataset.
- **`SyntheticData_Regression.ipynb`**: OAT of a linear model and a gaussion process on synthetic data.

<img title="a title" alt="Alt text" src="synthetic_full_result.svg">



## Citation

If you use this code, please cite:

```bibtex
@article{schmutz2026online,
  title={Online Active Testing: Adaptive Importance Sampling for Unbiased Risk Estimation in Data Streams},
  author={Schmutz, Hugo and Kadri, Hachem and Artières, Thierry},
  journal={Advances in Neural Information Processing Systems},
  year={2026}
}
```

## License

CC BY 4.0

## Contact

Hugo Schmutz: [Homepage](https://hugoschmutz.github.io/) 