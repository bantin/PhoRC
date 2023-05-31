import matplotlib.pyplot as plt
import matplotlib_ephys as mpe
import numpy as np
import seaborn as sns

from matplotlib_scalebar import scalebar
from mpl_toolkits.axes_grid1 import ImageGrid, make_axes_locatable

def plot_gridmaps(fig, mean_maps, depth_idxs,
    cmaps='viridis', vmin=None, vmax=None, zs=None, zlabels=None,
    powers=None,):

    # allow option to pass separate cmaps for each grid plot
    if not isinstance(cmaps, list):
        cmaps = len(mean_maps) * [cmaps]

    # Create an outer grid
    outer_grid = gridspec.GridSpec(1, len(mean_maps) + 1, width_ratios=[1]*len(mean_maps) + [0.05])
    
    # Calculate global min_val and max_val across all mean_maps if vmin and vmax are not provided
    if vmin is None:
        min_val = np.nanmin([np.nanmin(mean_map) for mean_map in mean_maps])
    else:
        min_val = vmin
    if vmax is None:
        max_val = np.nanmax([np.nanmax(mean_map) for mean_map in mean_maps])
    else:
        max_val = vmax

    for mean_idx, mean_map, cmap in zip(range(len(mean_maps)), mean_maps, cmaps):

        num_powers, _, _, num_planes = mean_map.shape
        num_planes_to_plot = len(depth_idxs)
        assert num_planes_to_plot <= num_planes

        # use subgrid for each ImageGrid
        subgrid = gridspec.GridSpecFromSubplotSpec(num_planes_to_plot, num_powers, subplot_spec=outer_grid[mean_idx], wspace=0.05, hspace=0.05)

        for j in range(num_planes_to_plot * num_powers):
            ax = plt.Subplot(fig, subgrid[j])
            fig.add_subplot(ax)
            
            ax.set_xticks([])
            ax.set_yticks([])
            
            row = j // num_powers
            col = j % num_powers
            
            im = ax.imshow(mean_map[col, :, :, depth_idxs[row]],
                           origin='lower', vmin=min_val, vmax=max_val, cmap=cmap)

            # optionally add labels
            if (zs is not None) and col == 0 and mean_idx == 0:
                ax.set_ylabel('%d ' % zs[depth_idxs[row]] + r'$\mu m $')
            elif (zlabels is not None) and col == 0 and mean_idx == 0:
                ax.set_ylabel(zlabels[row])

            # optionally add power label as white text on top of the image
            if powers is not None and row == 0:
                ax.annotate('%d mW' % powers[col], xy=(0.5, 1.05), xycoords='axes fraction',
                            horizontalalignment='right', verticalalignment='top')

        if mean_idx == len(mean_maps) - 1:
            colorbar_grid = gridspec.GridSpecFromSubplotSpec(num_planes // 2, 1, subplot_spec=outer_grid[-1],)
            cbar = plt.colorbar(im, cax=plt.subplot(colorbar_grid[0]))

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

    ax.plot(xs, ys, *args, **kwargs)

    if "label" in kwargs.keys():

        # remove duplicates
        handles, labels = plt.gca().get_legend_handles_labels()
        newLabels, newHandles = [], []
        for handle, label in zip(handles, labels):
            if label not in newLabels:
                newLabels.append(label)
                newHandles.append(handle)

        plt.legend(newHandles, newLabels)


def plot_current_traces(traces, msecs_per_sample=0.05,
                        time_cutoff=None, ax=None, stim_start_ms=5, stim_end_ms=10, plot_stim_lines=True,
                        scalebar=True, add_labels=False, IV_bar_length=None, box_aspect=2.0/3.0, **kwargs):
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

    # add scale bars and turn off spines
    if scalebar:
        mpe.plotting.draw_scale_bars(
            ax, style="paper", is_current=True, location="top", IV_bar_length=IV_bar_length)
        mpe.plotting.hide_spines(ax)

    ax.set_box_aspect(box_aspect)
    if add_labels:
        ax.set_xlabel('Time (ms)')
        ax.set_ylabel('Current (nA)')


def plot_gridmap_with_scalebars(map, targets, ax=None,
                                show_scalebar=True, show_colorbar=True,
                                scalebar_loc='lower right',
                                colorbar_position='right',
                                pixel_size=1.3,
                                **imshow_kwargs):
    if ax is None:
        ax = plt.gca()

    real_x = np.unique(targets[:, 0])
    real_y = np.unique(targets[:, 1])

    dx = (real_x[1]-real_x[0])/2.
    dy = (real_y[1]-real_y[0])/2.
    extent = [real_x[0]-dx, real_x[-1]+dx, real_y[0]-dy, real_y[-1]+dy]

    # plot the image
    im = ax.imshow(map, origin='lower', cmap='magma',
                   extent=extent, **imshow_kwargs)

    # add scale bar
    if show_scalebar:
        scb = scalebar.ScaleBar(1.3, 'um', frameon=True,
                                location=scalebar_loc, box_alpha=0.0, color='white')
        ax.add_artist(scb)

    # create an axes on the right side of ax. The width of cax will be 5%
    # of ax and the padding between cax and ax will be fixed at 0.05 inch.
    divider = make_axes_locatable(ax)
    cax = divider.append_axes(colorbar_position, size="5%", pad=0.05)

    if show_colorbar:
        cbar = plt.colorbar(im, cax=cax)
        if colorbar_position == 'left':
            cbar.ax.yaxis.set_ticks_position('left')
            cbar.ax.yaxis.set_tick_params(
                direction='out', labelleft=True, labelright=False)
    else:
        # make the cax invisible
        cax.set_visible(False)

    # remove axis ticks and labels
    ax.set_xticks([])
    ax.set_yticks([])

    # turn off frame around the image
    ax.set_frame_on(False)


def plot_maps_across_planes(axes, map, targets,
                            power_idx=-1, plane_idxs=[0], zs=None):

    # calculate min/max across all planes that will be plotted
    vmin = np.min(map[power_idx, :, :, plane_idxs])
    vmax = np.max(map[power_idx, :, :, plane_idxs])

    for i, (plane_idx, ax) in enumerate(zip(plane_idxs, axes)):
        if i == 0:
            show_colorbar = True
        else:
            show_colorbar = False
        if i == len(plane_idxs)-1:
            show_scalebar = True
        else:
            show_scalebar = False
        plot_gridmap_with_scalebars(map[power_idx, :, :, plane_idx],
                                    targets, ax=ax, show_scalebar=show_scalebar,
                                    show_colorbar=show_colorbar, vmin=vmin, vmax=vmax)

        if zs is not None:
            ax.set_ylabel('z = %d um' % zs[plane_idx])


def create_scatterplot(true_values, estimated_values, axis=None, pad=5, x_limits=None, y_limits=None, color='blue', alpha=0.2, s=0.2):

    assert len(true_values) == len(
        estimated_values), "True values and estimated values must have the same length."

    if axis is None:
        axis = plt.gca()

    # Create a scatterplot
    axis.scatter(true_values, estimated_values, color=color, alpha=alpha, s=s)

    # Make the plot square
    axis.set_aspect('equal', adjustable='box')

    if x_limits is None:
        minval = np.min(true_values)
        maxval = np.max(true_values)
        x_limits = [minval - pad, maxval + pad]
    if y_limits is None:
        minval = np.min(estimated_values)
        maxval = np.max(estimated_values)
        y_limits = [minval - pad, maxval + pad]

    axis.set_xlim(x_limits)
    axis.set_ylim(y_limits)

    # Draw the identity line
    axis.plot([x_limits[0], x_limits[1]], [y_limits[0], y_limits[1]],
              '--', label='Identity Line', color='grey')


def current_colors():
    return ['#00a651', '#7d49a5']


def before_after_colors():
    return ['#f68b1f', '#6dc8bf']


def plot_connections(weights, targets, ax=None, crop=0, s=10, **kwargs):

    connected_idxs = weights > 0

    # set background to gray
    if ax is None:
        ax = plt.gcf().add_subplot(111, aspect='equal')

    ax.set_box_aspect(1)
    ax.set_facecolor((0.7, 0.7, 0.7))

    # plot targeted locations as white circles
    vmin = None
    if 'vmin' in kwargs:
        vmin = kwargs['vmin']

    vmax = None
    if 'vmax' in kwargs:
        vmax = kwargs['vmax']

    ax.scatter(targets[:, 0], targets[:, 1], facecolors='none',
               edgecolors='white', linewidths=0.5, s=s)

    # fill in neurons we find as connected
    im = ax.scatter(targets[connected_idxs, 0], targets[connected_idxs, 1], c=weights[connected_idxs], cmap='magma',
                    edgecolors='white', linewidths=0.5, vmin=vmin, vmax=vmax, s=s)

    scb = scalebar.ScaleBar(1.0, 'um', frameon=False,
                            location='lower right', box_alpha=0.0)
    ax.add_artist(scb)

    # set axis limits to include a bit of padding
    minx = np.min(targets[:, 0])
    maxx = np.max(targets[:, 0])
    miny = np.min(targets[:, 1])
    maxy = np.max(targets[:, 1])
    minv = min(minx, miny)
    maxv = max(maxx, maxy)

    ax.set_xlim(minv + crop, maxv - crop)
    ax.set_ylim(minv + crop, maxv - crop)

    # turn off axis ticks and labels
    ax.set_xticks([])
    ax.set_yticks([])

    return im


# test plot_current_traces on random data
if __name__ == '__main__':
    traces = np.random.randn(10, 900)
    plot_current_traces(traces, msecs_per_sample=0.05)
    plt.show()
