import numpy as np
import matplotlib.pyplot as plt
import phorc.utils.grid_utils as grid_util

from circuitmap import NeuralDemixer
import phorc


def traces_tensor_to_map(tensor, idx_start=0, idx_end=-1):
    return np.nanmean(np.sum(tensor[...,idx_start:idx_end], axis=-1), axis=-1)

def plot_subtraction_comparison(raw_tensor, est_tensors, subtracted_tensors, demixed_tensors,
        powers, colors=['red', 'blue'], sort_by='raw', num_plots_per_power=30, z_idx=None,
        ):

    # see if we should restrict plot to only one plane
    npowers, nx, ny, nz, ntrials, ntimesteps = raw_tensor.shape
    if z_idx is None:
        z_idx = np.arange(nz)

    fig, axs = plt.subplots(nrows=num_plots_per_power * npowers, ncols=4,
        figsize=(10, num_plots_per_power*npowers*1.5),
        dpi=300, sharex=True, facecolor='white', sharey=False)

    # title each column
    titles = ['raw', 'est', 'subtracted', 'demixed']
    for j in range(4):
        axs[0,j].set_title(titles[j])

    # iterate over powers. Each power will get a separate block of rows
    for power_idx in range(npowers):
        axs[power_idx * num_plots_per_power, 0].annotate('%d mW' % powers[power_idx], xycoords='axes fraction',
                        xy=(-0.3,1.1))


        # get order from traces, depending on user-specified sorting
        raw_traces = raw_tensor[power_idx,:,:,z_idx,...].reshape(-1, ntrials, ntimesteps)

        if sort_by == 'raw':
            sort_map = traces_tensor_to_map(raw_traces)
        elif sort_by == 'subtracted':
            sort_map = traces_tensor_to_map(subtracted_tensors[0][power_idx,:,:,z_idx,...].reshape(-1, ntrials, ntimesteps))
        elif sort_by == 'demixed':
            sort_map = traces_tensor_to_map(demixed_tensors[0][power_idx,:,:,z_idx,...].reshape(-1, ntrials, ntimesteps))
        else:
            raise ValueError('Unknown argument for sorting')
        order = np.argsort(sort_map)[::-1]

        # set ylims for first two columns based on raw data
        raw_min = np.nanmin(raw_traces[:,0:500]) # ignore the last timesteps because of context from next trial
        raw_max = np.nanmax(raw_traces[:,0:500])

        # plot raw traces outside of the loop over subtraction methods
        for i in range(num_plots_per_power):
            row_idx = power_idx * num_plots_per_power + i
            idx = order[i]
            axs[row_idx, 0].plot(raw_traces[idx].T)
            axs[row_idx, 0].set_ylim(raw_min, raw_max)

            axs[row_idx, 0].annotate('%d' % row_idx, xycoords='axes fraction',
                        xy=(-0.3,0.5), rotation='vertical')


        # iterate over maps corresponding to different subtraction methods, plot each
        # in a different color
        for (est, subtracted_tensor, demixed_tensor, color) in zip(
            est_tensors, subtracted_tensors, demixed_tensors, colors):

            est_traces, subtracted_traces, demixed_traces = [x[power_idx,:,:,z_idx,...].reshape(-1, ntrials, ntimesteps)
                for x in [est, subtracted_tensor, demixed_tensor]]

            # set ylims for second two columns based on corrected data
            est_min = np.nanmin(subtracted_traces[:,:,0:400]) # ignore the last timesteps because of context from next trial
            est_max = np.nanmax(subtracted_traces[:,:,0:400])
 
            for i in range(num_plots_per_power):
                idx = order[i]
                row_idx = power_idx * num_plots_per_power + i

                # show photocurrent ests
                axs[row_idx, 1].plot(est_traces[idx].T, color=color, linewidth=0.5)
                axs[row_idx, 1].set_ylim(raw_min, raw_max)

                # show subtracted data
                axs[row_idx, 2].plot(subtracted_traces[idx].T, color=color, linewidth=0.5)
                axs[row_idx, 2].set_ylim(est_min, est_max)

                # show demixed
                axs[row_idx, 3].plot(demixed_traces[idx].T, color=color, linewidth=0.5)
                axs[row_idx, 3].set_ylim(est_min, est_max)


                for j in range(4):
                    axs[row_idx,j].axvline(x=100)
                    axs[row_idx,j].axvline(x=200)

    plt.tight_layout()
    return fig, axs


def plot_subtraction_by_power(pscs, ests, subtracted, demixed, powers,
        time=None, fig_kwargs=None, stim_start_ms=5, stim_end_ms=10):

    if fig_kwargs is None:
        fig_kwargs = dict(
            figsize=(9,3), 
            dpi=200,
            sharex=True
        )
    unique_powers = np.unique(powers)
    fig, axs = plt.subplots(nrows=len(unique_powers), ncols = 4, squeeze=False, **fig_kwargs)

    if time is None:
        time = np.arange(0,900) * 0.05
    for i in range(len(unique_powers)):
        these_trials = (powers == unique_powers[i])
        these_pscs = pscs[these_trials]
        these_ests = ests[these_trials]
        these_subtracted = subtracted[these_trials]
        these_demixed = demixed[these_trials]

        # order each by magnitude of photocurrent
        ordered_idx = np.argsort(np.sum(these_pscs, axis=-1))[::-1]
        these_pscs = these_pscs[ordered_idx]
        these_ests = these_ests[ordered_idx]
        these_demixed = these_demixed[ordered_idx]

        these_subtracted = these_subtracted[ordered_idx]

        axs[i,0].plot(time, these_pscs[0:20].T)
        axs[i,1].plot(time, these_ests[0:20].T)
        axs[i,2].plot(time, these_subtracted[0:20].T)
        axs[i,3].plot(time, these_demixed[0:20].T)

        # make ylim of first two plots match
        axs[i,1].set_ylim(axs[i,0].get_ylim())
        axs[i,3].set_ylim(axs[i,2].get_ylim())
        axs[i,0].set_ylabel('%d mW' % unique_powers[i])

        # add vertical lines for stimulus
        for j in range(4):
            axs[i,j].axvline(x=stim_start_ms, color='grey', linestyle='--')
            axs[i,j].axvline(x=stim_end_ms, color='grey', linestyle='--')

    labels = ['raw', 'est', 'subtracted', 'demixed']
    for i in range(4):
        axs[-1, i].set_xlabel('time (ms)')
        axs[0, i].set_title(labels[i])

    plt.tight_layout()
    return fig


def run_preprocessing_pipeline(pscs, powers, targets, stim_mat,
        demixer_path, estimate_args, subtraction_args, run_raw_demixed=False):

    # run phorc estimate
    est = phorc.estimate(pscs, **estimate_args, **subtraction_args)
    
    # load demixer checkpoint and demix
    subtracted = pscs - est
    demixer = NeuralDemixer(path=demixer_path, device='cpu')
    demixed = grid_util.denoise_pscs_in_batches(subtracted, demixer)

    # If run_raw_demixed is True, run the demixer on the raw data
    # and add to results
    if run_raw_demixed:
        raw_demixed = grid_util.denoise_pscs_in_batches(pscs, demixer)
        return dict(
            stim_mat=stim_mat,
            powers=powers, 
            targets=targets,

            # return traces matrices
            raw=pscs,
            est=est,
            subtracted=subtracted,
            demixed=demixed,
            raw_demixed=raw_demixed,
        )

    return dict(
        stim_mat=stim_mat,
        powers=powers, 
        targets=targets,

        # return traces matrices
        raw=pscs,
        est=est,
        subtracted=subtracted,
        demixed=demixed,
    )


def add_grid_results(results, idx_start=0, idx_end=-1):
    labels = ['raw', 'est', 'subtracted', 'demixed']
    if 'raw_demixed' in results:
        labels.append('raw_demixed')
    for label in labels:
        tensor = grid_util.make_psc_tensor_multispot(
            results[label],
            results['powers'], 
            results['targets'],
            results['stim_mat'] 
        )
        map = traces_tensor_to_map(tensor, idx_start=idx_start, idx_end=idx_end)

        # add tensor and map to results for 
        # convenient grid figures
        results[label + '_tensor'] = tensor
        results[label + '_map'] = map

    return results