from mpl_toolkits.axes_grid1 import ImageGrid
from mpl_toolkits.axes_grid1.axes_size import Fraction
import matplotlib as mpl
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf as pdf
import matplotlib.gridspec as gridspec
import seaborn as sns
import h5py
import subtractr
import subtractr.utils as utils

plt.rcParams.update({'font.size': 7, 'lines.markersize': np.sqrt(
    5), 'lines.linewidth': 0.5, 'lines.markeredgewidth': 0.25})
plt.rc('font', family='Helvetica')

mpl.rcParams['axes.spines.left'] = False
mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.spines.top'] = False
mpl.rcParams['axes.spines.bottom'] = False


def plot_gridmaps(fig, mean_maps, depth_idxs, cmaps='viridis', vmin=None, vmax=None, zs=None, zlabels=None, powers=None, show_annotations=True):
    if not isinstance(cmaps, list):
        cmaps = len(mean_maps) * [cmaps]

    max_num_powers = np.max([mean_map.shape[0] for mean_map in mean_maps])
    width_ratios = [mean_map.shape[0] /
                    max_num_powers for mean_map in mean_maps] + [0.05]
    outer_grid = gridspec.GridSpec(
        1, len(mean_maps) + 1, width_ratios=width_ratios)

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

        subgrid = gridspec.GridSpecFromSubplotSpec(
            num_planes_to_plot, num_powers, subplot_spec=outer_grid[mean_idx], wspace=0.05, hspace=0.05)

        for j in range(num_planes_to_plot * num_powers):
            ax = plt.Subplot(fig, subgrid[j])
            fig.add_subplot(ax)

            ax.set_xticks([])
            ax.set_yticks([])

            row = j // num_powers
            col = j % num_powers

            im = ax.imshow(mean_map[col, :, :, depth_idxs[row]],
                           origin='lower', vmin=min_val, vmax=max_val, cmap=cmap)

            if show_annotations:
                if (zs is not None) and col == 0 and mean_idx == 0:
                    ax.set_ylabel('%d ' % zs[depth_idxs[row]] + r'$\mu m $')
                elif (zlabels is not None) and col == 0 and mean_idx == 0:
                    ax.set_ylabel(zlabels[row])

                if powers is not None and row == 0 and num_powers == max_num_powers:
                    ax.annotate('%d mW' % powers[col], xy=(1.0, 1.0), xycoords='axes fraction',
                                horizontalalignment='right', verticalalignment='top',
                                color='white', fontsize=7)
            elif show_annotations and row == 0 and num_powers == max_num_powers:
                ax.annotate('%d mW' % powers[col], xy=(1.0, 1.0), xycoords='axes fraction',
                            horizontalalignment='right', verticalalignment='top',
                            color='white', fontsize=7)

        if mean_idx == len(mean_maps) - 1:
            colorbar_grid = gridspec.GridSpecFromSubplotSpec(
                np.maximum(num_planes // 2, 1), 1, subplot_spec=outer_grid[-1])
            cbar = plt.colorbar(im, cax=plt.subplot(colorbar_grid[0]))


def plot_summary_traces(raw, est, subtracted, stim_mat, demixed, num_to_plot=30, stim_end_idx=200):
    powers = np.max(stim_mat, axis=0)
    unique_powers = np.unique(powers)
    num_powers = len(unique_powers)

    fig = plt.figure(figsize=(6, 3), dpi=300, facecolor='white')
    gs_main = gridspec.GridSpec(num_powers, 2, width_ratios=[3, 1])

    data_to_plot = [-raw, -est, -subtracted, -subtracted]
    colors = sns.color_palette('magma', num_powers)
    box_aspect = 0.5
    for i, power, color in zip(range(len(unique_powers)), unique_powers, colors):
        power_idxs = np.where(powers == power)[0]

        # Plot N//2 traces with largest demixed response,
        # and N//2 traces with largest photocurrent estimates
        energy_est = np.sum(est[power_idxs, 0:stim_end_idx], axis=1)
        energy_demixed = np.sum(demixed[power_idxs, :], axis=1)

        idxs_to_plot_est = power_idxs[np.argsort(
            energy_est)[::-1][0:num_to_plot//2]]
        idxs_to_plot_demixed = power_idxs[np.argsort(
            energy_demixed)[::-1][0:num_to_plot//2]]

        idxs_to_plot = np.concatenate((idxs_to_plot_est, idxs_to_plot_demixed))

        gs_left = gridspec.GridSpecFromSubplotSpec(
            1, 3, subplot_spec=gs_main[i, 0])
        gs_right = gridspec.GridSpecFromSubplotSpec(
            1, 1, subplot_spec=gs_main[i, 1])
        gs_sub = [gs_left, gs_left, gs_left, gs_right]

        for j in range(4):
            axs = plt.subplot(gs_sub[j][0, j % 3])
            scalebar = True if j == 0 or j == 3 else False
            utils.plot_current_traces(
                traces=data_to_plot[j][idxs_to_plot, :], ax=axs, time_cutoff=38, box_aspect=box_aspect, scalebar=scalebar)

            if j != 3:
                axs.set_ylim([np.min(-raw[idxs_to_plot]),
                             np.max(-raw[idxs_to_plot])])
            else:
                axs.set_ylim([np.min(-subtracted[idxs_to_plot]),
                             np.max(-subtracted[idxs_to_plot])])

            axs.set_xticks([])
            axs.set_yticks([])
            axs.set_ylabel('')


def main(input_path, vmin=None, vmax=None, no_annotations=False):
    print('Loading data from %s' % input_path)
    R = utils.GridComparisonResults(ss_path=input_path)
    raw_map = R.results['singlespot']['raw_map']
    est_map = R.results['singlespot']['est_map']
    subtracted_map = R.results['singlespot']['subtracted_map']
    weights_map = R.get_singlespot_weights()
    powers = np.max(R.get_stim_mat(singlespot=True, multispot=False), axis=0)
    targets = R.results['singlespot']['targets']

    num_planes = raw_map.shape[-1]

    fig1 = plt.figure(figsize=(7, num_planes * 0.6),
                      dpi=300, facecolor='white')

    if vmax is None:
        vmax = np.max(subtracted_map)
    if vmin is None:
        vmin = 0

    plot_gridmaps(fig1, [raw_map, est_map, subtracted_map, weights_map[None, :]], np.arange(num_planes), cmaps='magma',
                  vmin=vmin, vmax=vmax, powers=np.unique(powers), zs=np.unique(targets[:, 2]), show_annotations=not no_annotations)
    dset_name = os.path.basename(input_path).split('.')[0]
    plt.savefig('%s_gridmap_supp.pdf' %
                dset_name, dpi=300, bbox_inches='tight')

    raw = R.get_singlespot_pscs(subtracted=False)
    est = R.get_singlespot_ests()
    subtracted = R.get_singlespot_pscs(subtracted=True)
    stim_mat = R.get_stim_mat(singlespot=True, multispot=False)
    demixed = R.get_singlespot_demixed(subtracted=True)

    plot_summary_traces(raw, est, subtracted, stim_mat, demixed)
    plt.savefig('%s_traces_supp.pdf' % dset_name, dpi=300, bbox_inches='tight')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process input file path.')
    parser.add_argument('--input_path', type=str,
                        help='path to the input file')
    parser.add_argument('--vmin', type=float,
                        help='minimum value for gridmap plot')
    parser.add_argument('--vmax', type=float,
                        help='maximum value for gridmap plot')
    parser.add_argument('--no_annotations', action='store_true',
                        help='disable "mW" annotations on gridmap plot')
    args = parser.parse_args()
    main(args.input_path, vmin=args.vmin, vmax=args.vmax,
         no_annotations=args.no_annotations)
