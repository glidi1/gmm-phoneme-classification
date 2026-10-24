# Mixture of Gaussians for Phoneme Classification

MSc coursework (Assignment 2: Unsupervised Learning) at Queen Mary University of London.

The notebook implements the EM algorithm for a Mixture of Gaussians (MoG) in NumPy and applies it to the Peterson and Barney vowel formant data. Each phoneme is modelled with its own MoG over the first two formants (F1 and F2), and the models are then used as a maximum likelihood classifier.

## What it covers

- Scatter plot of F1 against F2 for all 10 phonemes
- EM training of a MoG on a single phoneme, and how random initialisation changes the result between runs
- Training and saving MoGs for phonemes 1 and 2 with K = 3 and K = 6 components
- Classifying phonemes 1 and 2 by comparing the likelihood under each model
- Decision regions drawn over a grid of F1 and F2 values, for K = 3 and K = 6
- Adding a third feature (F1 + F2), which makes the covariance matrix singular and breaks training, then fixing it by adding a small value to the covariance diagonal

## Results

| Components | Classification accuracy |
|---|---|
| K = 3 | 0.951 |
| K = 6 | 0.964 |

K = 6 is slightly more accurate, but its decision boundary is more irregular and fits the training data more closely.

## Files

| File | Description |
|---|---|
| `wk8_Density_2024.ipynb` | Main notebook with code and saved outputs |
| `GMM_params_phoneme_01_k_03.npy` and the other three `GMM_params_*.npy` files | Trained MoG parameters for phoneme 1 and 2, with K = 3 and K = 6 |

Each parameter file holds a dictionary with the means (`mu`), covariances (`s`) and mixture weights (`p`):

```python
import numpy as np
params = np.load("GMM_params_phoneme_01_k_03.npy", allow_pickle=True).item()
print(params["mu"], params["s"], params["p"])
```

## Data

The notebook uses `PB_data.npy`, the Peterson and Barney vowel formant dataset (F0 to F3 and a phoneme id for each sample). It comes from the module materials, so it is not included here.


## Tech

Python, NumPy, SciPy, scikit-learn, Matplotlib, Jupyter

## Author

Elidian Terziu
[LinkedIn](https://linkedin.com/in/elidian-terziu)
