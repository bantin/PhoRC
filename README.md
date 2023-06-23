# **Pho**tocurrent **R**emoval with **C**onstraints (PhoRC)
Computational tools for removing direct photocurrent artifacts from intracellular recordings. This library implements PhoRC, described in more detail in our preprint:

> _Removing direct photocurrent artifacts in optogenetic connectivity mapping via constrained matrix factorization._ (2023) B. Antin\*, M. Sadahiro\*, M. A. Triplett, M. Gajowa, H. Adesnik, and L. Paninski

If you encounter bugs or issues using this library, please raise a Github issue or email ba2617@columbia.edu.

# Installation
This section coming soon :) 

# Low-rank based photocurrent subtraction
`phorc.estimate` runs a matrix-factorization based photocurrent estimation routine.

Required args: 

- `traces`: N x T array with recordings along the rows. For PhoRC to work well, we suggest using an overlapping trial structure, so that stimulation onset happens around frame 100 or so.

Optional args:
- `window_start`: This is the beginning index of the photocurrent integration window. Should be set to match stimulation onset. For example, in the PhoRC paper we align all traces so that stimulation begins 5ms into each trial. At a sampling rate of 20KHz, this results in using `window_start=100`. Default 100.


- `window_end`: This is the end of the photocurrent integration window. A good starting point is to match this to stimulus offset (e.g, frame 200 in our simulated data). If this results in overly-agressive photocurrent subtraction, this can be reduced to three milliseconds after stimulus onset. Default 200.

- `rank`: Number of temporal components used to estimate the photocurrent. Larger values of `rank` result in more aggressive subtraction at the cost of possibly subtracting PSCs. As a heuristic, we recommend using rank=1 when photocurrents are smaller than 0.1 nA, and rank=2 when photocurrents are larger than 0.1 nA. Default 1.

- `batch_size`: Size of batches used when splitting data for estimation. We recommend a batch size of 100-200 on real data. A value of `-1` processes all data in the same batch. Default -1.


