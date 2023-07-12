# **Pho**tocurrent **R**emoval with **C**onstraints (PhoRC)
Computational tools for removing direct photocurrent artifacts from intracellular recordings. This library implements PhoRC, described in more detail in our preprint:

> _Removing direct photocurrent artifacts in optogenetic connectivity mapping data via constrained matrix factorization._ (2023) B. Antin\*, M. Sadahiro\*, M. A. Triplett, M. Gajowa, H. Adesnik, and L. Paninski

If you encounter bugs or issues using this library, please raise a Github issue or email ba2617@columbia.edu.

# Quick start
The fastest way to see PhoRC in action is to check out our [demo notebook](https://colab.research.google.com/github/bantin/PhoRC/blob/master/examples/phorc_demo.ipynb) which downloads an example ChroME2f dataset and runs photocurrent subtraction.

For running on your own data, you'll likely want to install PhoRC locally. We recommend performing installation in a fresh conda environment. PhoRC's main dependency is [JAX](https://github.com/google/jax). GPU support is not required. PhoRC has been tested with JAX until 0.4.0. Once JAX is installed, clone this repository and install it with pip:

```
git clone https://github.com/bantin/PhoRC
pip install ./phorc
```

# Low-rank based photocurrent subtraction
`phorc.estimate` runs a matrix-factorization based photocurrent estimation routine.

Required args: 

- `traces`: N x T array with recordings along the rows. For PhoRC to work well, we suggest using an overlapping trial structure, so that stimulation onset happens around frame 100 or so.

Optional args:
- `window_start`: This is the beginning index of the photocurrent integration window. Should be set to match stimulation onset. For example, in the PhoRC paper we align all traces so that stimulation begins 5ms into each trial. At a sampling rate of 20KHz, this results in using `window_start=100`. Default 100.


- `window_end`: This is the end of the photocurrent integration window. A good starting point is to match this to stimulus offset (e.g, frame 200 in our simulated data). If this results in overly-agressive photocurrent subtraction, this can be reduced to three milliseconds after stimulus onset. Default 200.

- `rank`: Number of temporal components used to estimate the photocurrent. Larger values of `rank` result in more aggressive subtraction at the cost of possibly subtracting PSCs. As a heuristic, we recommend using rank=1 when photocurrents are smaller than 0.1 nA, and rank=2 when photocurrents are larger than 0.1 nA. Default 1.

- `batch_size`: Size of batches used when splitting data for estimation. We recommend a batch size of 100-200 on real data. A value of `-1` processes all data in the same batch. Default -1.


