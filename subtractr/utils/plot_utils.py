import matplotlib.pyplot as plt
import numpy as np


def plot_multi_means(fig, mean_maps, depth_idxs,
                     roi_bounds=None,
                     zs=None, zlabels=None, powers=None, map_names=None, cmaps='viridis',
                     vranges=None, cbar_labels=None, show_powers=None):
    if show_powers is None:
        show_powers = True * [len(mean_maps)]

    # allow option to pass separate cmaps for each grid plot
    if not isinstance(cmaps, list):
        cmaps = len(mean_maps) * [cmaps]

    for mean_idx, mean_map, cmap in zip(
            range(len(mean_maps)), mean_maps, cmaps):

        num_powers, _, _, num_planes = mean_map.shape
        num_planes_to_plot = len(depth_idxs)
        assert num_planes_to_plot <= num_planes

        # Create a new grid for each mean map
        subplot_args = int("1" + str(len(mean_maps)) + str(mean_idx + 1))
        ax_curr = plt.subplot(subplot_args)

        if powers is not None and map_names is not None:
            ax_curr.set_title(map_names[mean_idx], y=1.08)

        plt.axis('off')

        grid = ImageGrid(fig, subplot_args,  # similar to subplot(111)
                         # creates 2x2 grid of axes
                         nrows_ncols=(num_planes_to_plot, num_powers),
                         axes_pad=0.05,  # pad between axes in inch.
                         cbar_mode='single',
                         cbar_pad=0.2
                         )
        if vranges is not None:
            min_val, max_val = vranges[mean_idx]
        else:
            min_val = np.nanmin(mean_map)
            max_val = np.nanmax(mean_map)

        for j, ax in enumerate(grid):
            row = j // num_powers
            col = j % num_powers
            ax.set_xticks([])
            ax.set_yticks([])
#             ax.set_frame_on(False)

            # optionally add labels
            if (zs is not None) and col == 0 and mean_idx == 0:
                ax.set_ylabel('%d ' % zs[depth_idxs[row]] + r'$\mu m $')
            elif (zlabels is not None) and col == 0 and mean_idx == 0:
                ax.set_ylabel(zlabels[row])

            if powers is not None and row == num_planes_to_plot - 1 and show_powers[mean_idx]:
                ax.set_xlabel('%d mW' % powers[col], rotation=70)

            im = ax.imshow(mean_map[col, :, :, depth_idxs[row]],
                           origin='lower', vmin=min_val, vmax=max_val, cmap=cmap)

            cbar = grid[0].cax.colorbar(im)

            if roi_bounds is not None:
                roi_curr = roi_bounds[mean_idx]
                rect = patches.Rectangle((50, 100), 40,
                                         30, linewidth=1, edgecolor='r', facecolor='none')

                # Add the patch to the Axes
                ax.add_patch(rect)

        if cbar_labels is not None:
            cbar.set_label(cbar_labels[mean_idx], rotation=90, loc='top')

def plot_spike_inference_with_waveforms(den_psc, stim, I, model_state, waveforms=None, latencies=None,
                                        spike_thresh=0.01, save=None, ymax=None, n_plots=15, num_trials=30,
                                        weights=None, col_width=10.5, row_height=0.6, order=None,
                                        title=None, raw_psc=None, fontsize=14):
    N = stim.shape[0]
    K = den_psc.shape[0]
    trial_len = 900
    normalisation_factor = np.max(np.abs(den_psc))
    trace_linewidth = 0.65
    ymax = 1.05
    ymin = -0.05 * ymax

    mu = model_state['mu']
    mu /= (np.max(mu) + 1e-9)
    lam = model_state['lam']
    z = model_state['z']

    fig = plt.figure(figsize=(col_width, row_height *
                     n_plots * 1.5), dpi=300, facecolor='white')

    # one columns for inferred waveforms
    powers = np.unique(I)
    width_ratios = np.zeros(len(powers) + 1)
    width_ratios[0:len(powers)] = 1
    width_ratios[-1] = n_plots
    gs = fig.add_gridspec(ncols=len(powers) + 1, nrows=n_plots,
                          hspace=0.5, wspace=0.05, width_ratios=width_ratios)

    if order is None:
        order = np.argsort(mu)[::-1]

    for m in range(n_plots):
        n = order[m]

        # spike predictions
        ax = fig.add_subplot(gs[m, -1])

        if title is not None and m == 0:
            plt.title(title, fontsize=fontsize, y=1.5)

        trials_per_power = num_trials // len(powers)
        stim_locs = np.array([])
        for pwr in powers:
            stim_locs = np.concatenate(
                [stim_locs, np.where(stim[n] == pwr)[0][:trials_per_power]])

        stim_locs = stim_locs.astype(int)
        this_y_psc = den_psc[stim_locs].flatten()/normalisation_factor
        n_repeats = np.min([len(stim_locs), num_trials])
        trial_breaks = np.arange(0, trial_len * n_repeats + 1, trial_len)

        if raw_psc is not None:
            this_y_psc_raw = raw_psc[stim_locs].flatten()/normalisation_factor

        plt.xlim([0, trial_len*n_repeats])

        # if we have ground truth weights
        if weights is None:
            trace_col = 'k'
        else:
            trace_col = 'k' if weights[n] != 0 else 'gray'

        facecol = 'firebrick'
        for tb in range(len(trial_breaks) - 1):
            if tb > 0:
                plt.plot([trial_breaks[tb], trial_breaks[tb]],
                         [ymin, ymax], '--', color=trace_col)

            ax.fill_between(np.arange(trial_len * tb, trial_len * (tb + 1)), ymin * np.ones(trial_len), ymax * np.ones(trial_len), facecolor=facecol,
                            edgecolor='None', alpha=lam[n, stim_locs][tb] * mu[n], zorder=-5)

            if z[stim_locs][tb] != 0:
                plt.plot(trial_len * (tb + 0.5), 0.75 * ymax, marker='*',
                         markerfacecolor='b', markeredgecolor='None', markersize=6)

            # Plot power changes
            if (m == 0) and (I[stim_locs][tb] != I[stim_locs][tb-1]):
                plt.text(trial_breaks[tb], 1.1 * ymax, '%i mW' %
                         I[stim_locs][tb], fontsize=fontsize-2)

        plt.plot(this_y_psc, color=trace_col, linewidth=trace_linewidth)
        if raw_psc is not None:
            plt.plot(this_y_psc_raw, color='blue',
                     linewidth=trace_linewidth, alpha=0.5)

        for loc in ['top', 'right', 'left', 'bottom']:
            plt.gca().spines[loc].set_visible(False)
        plt.xticks([])
        plt.yticks([])
        plt.ylim([ymin, ymax])
#         plt.ylabel(m+1, fontsize=fontsize-1, rotation=0, labelpad=15, va='center')

        ax.set_rasterization_zorder(-2)

    if waveforms is not None:
        ### Inferred PSC waveforms ###
        waveform_colors = ['blue', 'green', 'purple', 'red']

        for m in range(n_plots):
            n = order[m]

            for power_idx, power in enumerate(powers):
                ax = fig.add_subplot(gs[m, power_idx])

                plt.plot(waveforms[power_idx, n, :]/normalisation_factor,
                         color=waveform_colors[power_idx], linewidth=trace_linewidth)

                # draw vertical line at inferred latency, first convert to index
                sample_khz = 20
                samples_per_sec = sample_khz * 1e3
                secs_per_sample = 1 / samples_per_sec
                msecs_per_sample = secs_per_sample * 1e3
                plt.axvline(x=(latencies[power_idx, n] / msecs_per_sample),
                            color=waveform_colors[power_idx],
                            linewidth=trace_linewidth, linestyle='-.')

                for loc in ['top', 'right', 'left', 'bottom']:
                    plt.gca().spines[loc].set_visible(False)
                plt.xticks([])
                plt.yticks([])
                plt.ylim([ymin, ymax])

    return fig

def plot_collection(ax, xs, ys, *args, **kwargs):

  ax.plot(xs,ys, *args, **kwargs)

  if "label" in kwargs.keys():

    #remove duplicates
    handles, labels = plt.gca().get_legend_handles_labels()
    newLabels, newHandles = [], []
    for handle, label in zip(handles, labels):
      if label not in newLabels:
        newLabels.append(label)
        newHandles.append(handle)

    plt.legend(newHandles, newLabels)

def plot_current_traces(traces, msecs_per_sample=0.05,
    time_cutoff=None, ax=None, stim_start_ms=5, stim_end_ms=10, plot_stim_lines=True, **kwargs):
    """
    Plot a collection of current traces.
    params:
        traces: a 2D array of current traces
        msecs_per_sample: the number of milliseconds per sample
        time_cutoff: the number of milliseconds to plot
        ax: the axis to plot on
        stim_start_ms: the start of the stimulus in milliseconds
        stim_end_ms: the end of the stimulus in milliseconds
        plot_stim_lines: whether to plot the stimulus lines
    """
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = None

    if time_cutoff is None:
        time_cutoff = traces.shape[1] * msecs_per_sample

    ax.plot(np.arange(0, time_cutoff, msecs_per_sample),
        traces[:, :int(time_cutoff / msecs_per_sample)].T, **kwargs)

    if plot_stim_lines:
        ax.axvline(x=stim_start_ms, color='grey', linestyle='--')
        ax.axvline(x=stim_end_ms, color='grey', linestyle='--')

    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Current (nA)')


# test plot_current_traces on random data
if __name__ == '__main__':
    traces = np.random.randn(10, 900)
    plot_current_traces(traces, msecs_per_sample=0.05)
    plt.show()

