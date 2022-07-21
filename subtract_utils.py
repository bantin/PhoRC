import numpy as np
import matplotlib.pyplot as plt
import grid_utils as util

from circuitmap import NeuralDemixer
import subtractr


def traces_tensor_to_map(tensor):
    return np.nanmean(np.sum(tensor, axis=-1), axis=-1)


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

def run_subtraction_pipeline(pscs, powers, targets, stim, demixer_checkpoint, no_op=False, **run_kwargs):
    # Run subtraction on all PSCs
    if no_op:
        est = np.zeros_like(pscs)
    else:
        est = subtractr.estimate_photocurrents_baseline(pscs, powers, **run_kwargs)
    subtracted = pscs - est

    # load demixer checkpoint and demix
    demixer = NeuralDemixer(path=demixer_checkpoint, device='cpu')
    demixed = util.denoise_pscs_in_batches(subtracted, demixer)

    # convert to tensors for easier plotting
    raw_pscs_tensor = util.make_psc_tensor_multispot(pscs, powers, targets, stim)
    est_pscs_tensor = util.make_psc_tensor_multispot(est, powers, targets, stim)
    subtracted_pscs_tensor = util.make_psc_tensor_multispot(subtracted, powers, targets, stim)
    demixed_pscs_tensor = util.make_psc_tensor_multispot(demixed, powers, targets, stim)

    # make plot of spatial maps
    mean_map = traces_tensor_to_map(raw_pscs_tensor)
    mean_map_subtracted = traces_tensor_to_map(subtracted_pscs_tensor)
    mean_map_demixed = traces_tensor_to_map(demixed_pscs_tensor)

    return dict(
        # return traces matrices
        raw_matrix=pscs,
        est_matrix=est,
        subtracted_matrix=subtracted,
        demixed_matrix=demixed,
        
        # return traces tensors
        raw_tensor=raw_pscs_tensor,
        est_tensor=est_pscs_tensor,
        subtracted_tensor=subtracted_pscs_tensor,
        demixed_tensor=demixed_pscs_tensor,
        # return grid maps
        raw_map=mean_map,
        subtracted_map=mean_map_subtracted,
        demixed_map=mean_map_demixed

    )

def make_subtraction_figs_singlespot(pscs, I, L, dataset_name, demixer_checkpoint):
    y_raw = pscs.sum(1)
    grid_mean, _, num_stims = util.make_suff_stats(y_raw, I, L)
    num_powers, num_xs, num_ys, num_zs = grid_mean.shape


    # Run subtraction on all PSCs
    est = subtractr.estimate_photocurrents(pscs, I, separate_by_power=True)
    subtracted = pscs - est

    # load demixer checkpoint and demix
    demixer = NeuralDemixer(path=demixer_checkpoint)
    demixed = util.denoise_pscs_in_batches(subtracted, demixer)

    # convert to tensors for easier plotting
    raw_pscs_tensor = util.make_psc_tensor(pscs, I, L)
    est_pscs_tensor = util.make_psc_tensor(est, I, L)
    subtracted_pscs_tensor = util.make_psc_tensor(subtracted, I, L)
    demixed_pscs_tensor = util.make_psc_tensor(demixed, I, L)

    # make plot of spatial maps
    mean_map = traces_tensor_to_map(raw_pscs_tensor)
    mean_map_subtracted = traces_tensor_to_map(subtracted_pscs_tensor)
    mean_map_demixed = traces_tensor_to_map(demixed_pscs_tensor)

    fig2 = plt.figure(figsize=(6 * num_powers, num_zs), dpi=300, facecolor='white')

    util.plot_multi_means(fig2,
        [mean_map, mean_map_subtracted, mean_map_demixed], np.arange(num_zs),
    #     map_names=['subtracted'],
        cmaps=['magma', 'magma', 'magma'],
        # cbar_labels=['EPSQ (nC)'],
        # zlabels=['subtr', 'demix'],
        map_names=['raw', 'subtr', 'demix'],
        # vranges=[(0,5), (0,5.0), (0,5.0)],
        powers=np.unique(I))

    # # make plot of traces
    # max_raw = np.max(pscs[:,0:500])
    # min_raw = np.min(pscs[:,0:500])
    # max_subtracted = np.max(subtracted[:,0:500])
    # min_subtracted = np.min(subtracted[:,0:500])

    fig3, axs = plot_subtraction_comparison(raw_pscs_tensor,
        [est_pscs_tensor],
        [subtracted_pscs_tensor,],
        [demixed_pscs_tensor],
        powers=np.unique(I),
    # ylims=[(min_raw, max_raw,), (min_raw, max_raw),
    # (min_subtracted, max_subtracted), (min_subtracted, max_subtracted)],
    # z_idx=-1,
    # power_idx=-2,
    # override_sharey=False)
    )
    axs[0,0].set_ylabel('Current (nA)')
    # plt.tight_layout()