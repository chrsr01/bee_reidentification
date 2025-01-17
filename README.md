# bee-abdomen-reidentification

## Excerpt of the Paper Introduction

Animal reidentification, a growing field in computer vision, focuses on identifying and tracking individual animals over time and across locations. This technology supports wildlife conservation, behavioral studies, and ecological monitoring by providing scalable, non-invasive alternatives to traditional methods like tagging and manual observation. Recent advances in deep learning, particularly convolutional neural networks (CNNs), have significantly improved the accuracy of animal reidentification systems.

This project reproduces the work of Chan et al. (2022) on reidentifying honeybees using self-supervised learning. Their approach embedded bee images in a 128-dimensional Euclidean space using a CNN trained with triplet loss and semi-hard online mining. They pre-trained the network on data assuming unique bee identities within short time frames, capturing variations like lighting and pose. While the original study highlighted the promise of self-supervision, it did not explore the impact of selecting triplets based on temporal differences. This repository extends their findings to investigate this aspect.


## Project Organization

```
├── README.md          <- The top-level README for developers using this project.

├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── models             <- Model scripts
│
├── notebooks          <- Jupyter notebooks.
│
├── setup.cfg          <- Configuration file for flake8
│
└── bee-abomden-reidentification   <- Source code for use in this project.
    │
    ├── __init__.py             <- Makes bee-abomden-reidentification a Python module
    │
    ├── config.py               <- Store useful variables and configuration
    │
    ├── dataset.py              <- Scripts to download or generate data
    │
    ├── features.py             <- Code to create features for modeling
    │
    ├── modeling                
    │   ├── __init__.py 
    │   ├── predict.py          <- Code to run model inference with trained models          
    │   └── train.py            <- Code to train models
    │
    └── plots.py                <- Code to create visualizations
```

--------

