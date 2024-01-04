import phorc
import h5py
import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
import phorc.utils as utils
import os
import seaborn as sns
import omegaconf
import hydra

from phorc.utils import plot_before_after_spatial, compute_waveforms_by_power
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib_scalebar import scalebar
from matplotlib.backends.backend_pdf import PdfPages

plt.rcParams.update({'font.size': 7, 'lines.markersize': np.sqrt(5), 'lines.linewidth': 0.5, 'lines.markeredgewidth': 0.25})
plt.rc('font', family='Helvetica')

@hydra.main()
def main(cfg):
        with h5py.File(cfg.dataset_path) as f:
                pscs = np.array(f['pscs']).T
                stim_mat = np.array(f['stimulus_matrix']).T
                targets = np.array(f['targets']).T
                powers = np.max(stim_mat, axis=0)

        # get rid of any trials where we didn't actually stim
        good_idxs = (powers > 0)
        pscs = pscs[good_idxs, :]
        stim_mat = stim_mat[:, good_idxs]
        powers = powers[good_idxs]

        # create pdf for saving figures
        file_output_path = os.path.join(cfg.output_path, '%s_batch_size_sweep.pdf' % os.path.basename(cfg.dataset_path))
        pdf = PdfPages(file_output_path)

        for batch_size in cfg.batch_sizes:
                ests = phorc.estimate(pscs, window_start_idx=100, window_end_idx=200, batch_size=batch_size, rank=cfg.rank)
                fig = utils.plot_subtraction_by_power(pscs, ests, pscs - ests, np.zeros_like(pscs), powers, fig_kwargs={'sharey': 'row', 'dpi': 300})
                fig.suptitle(f'Batch size = {batch_size}')
                pdf.savefig(fig)

        pdf.close()

if __name__ == '__main__':

        # make hydra config
        cfg = omegaconf.OmegaConf.create({
            'dataset_path': '', 
            'output_path': '',
            'batch_sizes': [20, 50, 200, 2000],
            'rank': 2,
        })

        cfg = omegaconf.OmegaConf.merge(cfg, omegaconf.OmegaConf.from_cli())
        print(cfg)
        main(cfg)





