import numpy as np
import circuitmap as cm
from circuitmap import NeuralDemixer
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge


def _get_pairwise_dist_xy(tars):
    return np.array([[np.sqrt(np.sum(np.square(tar1 - tar2))) for tar1 in tars] for tar2 in tars])


def _get_pairwise_adjacency_z(tars, planes):
    plane_representation = np.array(
        [np.where(tar == planes)[0][0] for tar in tars])
    adj = np.array([[
        np.abs(tar1 - tar2) <= 1 for tar1 in plane_representation]
        for tar2 in plane_representation
    ]).astype(int)
    return adj


def _get_cluster_reps_pixelbased(clusters, targets, img):
    planes = np.unique(targets[:, -1])
    n_clusters = len(clusters)
    cluster_reps = [None for _ in range(n_clusters)]
    for i in range(n_clusters):
        pixel_brightness = []
        for cl in clusters[i]:
            tar = targets[cl].astype(int)
            depth_indx = np.where(tar[-1] == planes)[0][0]
            pixel_brightness += [img[0][depth_indx][tar[0], tar[1]]]
        cluster_reps[i] = clusters[i][np.argmax(pixel_brightness)]
    return cluster_reps


def compute_ridge_waveforms(psc, model_state, stim_matrix):
    cnx = np.where(model_state['mu'])[0]
    locs = np.unique(np.concatenate(
        np.array([np.where(stim_matrix[n])[0] for n in cnx])))
    lr = Ridge(fit_intercept=False, alpha=1e-3)
    lr.fit(model_state['lam'][cnx][:, locs].T, psc[locs])
    return lr.coef_.T


def merge_duplicates(psc, stim_matrix, model_state, targets, img, mse_threshold=0.1, dist_threshold=15):
    planes = np.unique(targets[:, -1])
    weights = model_state['mu']
    found_cnx = np.where(weights)[0]
    n_cnx = len(found_cnx)
    waveforms = compute_ridge_waveforms(psc, model_state, stim_matrix)
    pairwise_errs = np.array([
        [
            np.sum(np.square(waveforms[cnx1] - waveforms[cnx2])) for cnx1 in range(n_cnx)
        ] for cnx2 in range(n_cnx)
    ])
    pairwise_adj = _get_pairwise_adjacency_z(targets[found_cnx][:, -1], planes)
    # close in xy and lie on adjacent planes
    pairwise_close = (_get_pairwise_dist_xy(
        targets[found_cnx][:, :2]) < dist_threshold) * pairwise_adj
    pairwise_duplicates = (pairwise_errs < mse_threshold) * pairwise_close
    clusters = [list(x) for x in set([tuple(found_cnx[np.where(row)[0]].tolist())
                                      for row in pairwise_duplicates])]  # extract duplicate clusters
    cluster_reps = _get_cluster_reps_pixelbased(
        clusters, targets, img)  # select cluster representatives
    return cluster_reps


def lookup(coords, arr):
    return np.intersect1d(*[np.where(arr[:, i] == coords[i])[0] for i in range(2)])[0]


def plot_spike_inference_comparison(den_pscs, stim_matrices, models, spks=None, titles=None, save=None,
                                    ymax=1.1, n_plots=15, max_trials_to_show=30, col_widths=None, row_height=0.6, order=None, trial_len=900):
    if col_widths is None:
        col_widths = 7.5 * np.ones(len(models))

    N = stim_matrices[0].shape[0]
    Is = [np.array([np.unique(stim[:, k])[1] for k in range(stim.shape[1])])
          for stim in stim_matrices]
    ncols = len(models)

    fig = plt.figure(figsize=(np.sum(col_widths), row_height * n_plots * 1.5))
    gs = fig.add_gridspec(ncols=ncols, nrows=n_plots, hspace=0.5,
                          wspace=0.05, width_ratios=col_widths/col_widths[0])

    normalisation_factor = np.max(np.abs(np.vstack(den_pscs)))
    mu_norm = np.max(np.abs([model.state['mu'] for model in models]))
    ymin = -0.05 * ymax

    trace_linewidth = 0.65

    if order is None:
        order = np.argsort(models[0].state['mu'])[::-1]

    for col in range(ncols):
        for m in range(n_plots):
            n = order[m]

            # spike predictions
            ax = fig.add_subplot(gs[m, col])
            if m == 0 and titles is not None:
                plt.title(titles[col], fontsize=fontsize, y=1.5)

            powers = np.unique(Is[col])
            trials_per_power = max_trials_to_show // len(powers)
            stim_locs = np.array([])
            for pwr in powers:
                stim_locs = np.concatenate([stim_locs, np.where(
                    stim_matrices[col][n] == pwr)[0][:trials_per_power]])

            stim_locs = stim_locs.astype(int)
            this_y_psc = den_pscs[col][stim_locs].flatten() / \
                normalisation_factor
            n_repeats = np.min([len(stim_locs), max_trials_to_show])
            trial_breaks = np.arange(0, trial_len * n_repeats + 1, trial_len)

            plt.xlim([0, trial_len*n_repeats])
            facecol = 'firebrick'
            model = models[col]
            lam = model.state['lam']
            K = lam.shape[1]
            mu = model.state['mu'].copy()
            trace_col = 'k' if mu[n] != 0 else 'gray'

            if 'z' in model.state.keys():
                z = model.state['z']
            else:
                z = np.zeros(K)

            for tb in range(len(trial_breaks) - 1):
                if tb > 0:
                    plt.plot([trial_breaks[tb], trial_breaks[tb]],
                             [ymin, ymax], '--', color=trace_col)

                ax.fill_between(np.arange(trial_len * tb, trial_len * (tb + 1)), ymin * np.ones(trial_len), ymax * np.ones(trial_len), facecolor=facecol,
                                edgecolor='None', alpha=lam[n, stim_locs][tb] * 0.5, zorder=-5)

                # Plot power changes
                if (m == 0) and (Is[col][stim_locs][tb] != Is[col][stim_locs][tb-1]):
                    plt.text(trial_breaks[tb], 1.1 * ymax, '%i mW' %
                             Is[col][stim_locs][tb], fontsize=fontsize-2)

                if z[stim_locs][tb] != 0:
                    plt.plot(trial_len * (tb + 0.5), 0.7 * ymax, marker='*',
                             markerfacecolor='b', markeredgecolor='None', markersize=6)

            plt.plot(this_y_psc, color=trace_col, linewidth=trace_linewidth)

            for loc in ['top', 'right', 'left', 'bottom']:
                plt.gca().spines[loc].set_visible(False)
            plt.xticks([])
            plt.yticks([])
            plt.ylim([ymin, ymax])

            if col == 0:
                label_col = 'k'
                plt.ylabel(m+1, fontsize=fontsize-1, rotation=0,
                           labelpad=15, va='center', color=label_col)

                fig.supylabel('Neuron', fontsize=fontsize, x=0.09)  # x=0.0825)
            ax.set_rasterization_zorder(-2)

    if save is not None:
        plt.savefig(save, format='png', bbox_inches='tight',
                    dpi=400, facecolor='white')
