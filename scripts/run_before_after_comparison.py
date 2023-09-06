import numpy as np
from mpl_toolkits.axes_grid1 import ImageGrid
import matplotlib.pyplot as plt
from circuitmap import NeuralDemixer
import circuitmap as cm
import h5py
import os
import phorc
import phorc.utils as utils
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf, DictConfig
import hydra

plt.rcParams['figure.dpi'] = 300



def split_results_dict(results):
    """Split the results dict into two dicts, one for single target and one for multi-target trials.

    Parameters
    ----------
    results : dict
        Dictionary of results from running the preprocessing pipeline.
        Expected to contain the keys: stim_mat, pscs, demixed, subtracted, est, raw, raw_demixed
    
    Returns
    -------
    results : dict
        Dictionary of results from running the preprocessing pipeline.
        Contains the keys: targets, singlespot, multispot
    """
    stim_mat, raw, demixed, subtracted, est, raw_demixed = results['stim_mat'], results['raw'], results[
        'demixed'], results['subtracted'], results['est'], results['raw_demixed']
    targets = results['targets']
    unique_target_counts = np.unique(np.sum(stim_mat > 0, axis=0))
    max_target_count = unique_target_counts[-1]
    min_target_count = unique_target_counts[0]

    results_new = {}

    # get indices for singlespot and multispot trials
    singlespot_idxs = np.sum(stim_mat > 0, axis=0) == min_target_count
    multispot_idxs = np.sum(stim_mat > 0, axis=0) == max_target_count

    # fill in singlespot results
    if min_target_count == 1:
        results_new['singlespot'] = {}
        these_idxs = np.sum(stim_mat > 0, axis=0) == min_target_count
        results_new['singlespot']['stim_mat'] = stim_mat[:, these_idxs]
        results_new['singlespot']['demixed'] = demixed[these_idxs]
        results_new['singlespot']['subtracted'] = subtracted[these_idxs]
        results_new['singlespot']['est'] = est[these_idxs]
        results_new['singlespot']['raw'] = raw[these_idxs]
        results_new['singlespot']['raw_demixed'] = raw_demixed[these_idxs]

    # fill in multispot results if available
    if max_target_count > 1:
        results_new['multispot'] = {}
        these_idxs = np.sum(stim_mat > 0, axis=0) == max_target_count
        results_new['multispot']['stim_mat'] = stim_mat[:, these_idxs]
        results_new['multispot']['demixed'] = demixed[these_idxs]
        results_new['multispot']['subtracted'] = subtracted[these_idxs]
        results_new['multispot']['est'] = est[these_idxs]
        results_new['multispot']['raw'] = raw[these_idxs]
        results_new['multispot']['raw_demixed'] = raw_demixed[these_idxs]

    return results_new

def save_dicts_to_hdf5(hdf5_filename, mode='a', **dicts):
    """
    Save multiple dictionaries containing numpy arrays to an HDF5 file.

    Parameters:
    - hdf5_filename: name of the HDF5 file to save to.
    - dicts: one or more dictionaries with string keys and numpy array values.
    """
    with h5py.File(hdf5_filename, mode) as f:
        for dict_name, dictionary in dicts.items():
            if dict_name in f:
                del f[dict_name]
            grp = f.create_group(dict_name)
            for key, value in dictionary.items():
                if isinstance(value, np.ndarray):
                    # Check if the dtype is "object" and skip if it is
                    if value.dtype == 'object':
                        print(f"Warning: {key} in {dict_name} has dtype 'object'. Skipping...")
                        continue
                    if key in grp:
                        del grp[key]
                    grp.create_dataset(key, data=value)
                else:
                    print(f"Warning: {key} in {dict_name} is not a numpy array. Skipping...")

@hydra.main(version_base=None, config_path="conf", config_name="before_after_config")
def main(cfg: DictConfig):

    # extract arguments by group
    estimate_args = cfg["estimate"]
    phorc_args = cfg["phorc"]
    caviar_args = cfg["caviar"]
    
    dset_name = os.path.basename(cfg.dataset_path).split('.')[0]

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

    # Run the preprocessing pipeline creating demixed data both
    # with and without the subtraction step.
    results = utils.run_preprocessing_pipeline(
        pscs, powers, targets, stim_mat, cfg.demixing.demixer_path,
        estimate_args, phorc_args, run_raw_demixed=True,
    )

    # split the results dict into single and multi target count groups
    results = split_results_dict(results)
    for target_count_label in ['singlespot', 'multispot']:
        if target_count_label not in results:
            continue
        for psc_label, output_label in zip(['demixed', 'raw_demixed'], ['subtraction_on', 'subtraction_off']):

            print('Running caviar with %s' % output_label)
            pscs_curr = results[target_count_label][psc_label]
            stim_mat_curr = results[target_count_label]['stim_mat']
            N = stim_mat_curr.shape[0]
            model = cm.Model(N)
            model.fit(pscs_curr,
                stim_mat_curr,
                method='caviar',
                fit_options = caviar_args,
            )
            results[target_count_label]['model_state_' + output_label] = model.state



    # save args used to get the result
    outpath = os.path.join(cfg.save_path, dset_name + '_before_after_comparison.npz')
    argstr = OmegaConf.to_yaml(cfg)
    np.savez_compressed(outpath, **results, args=argstr)

if __name__ == '__main__':
    main()