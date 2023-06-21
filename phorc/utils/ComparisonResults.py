import numpy as np
from subtractr.utils.grid_utils import sort_results
from subtractr.utils.subtract_utils import add_grid_results

def _fix_results_dict(results):
    # Due to some numpy weirdness, the results dict contains
    # lots of dictionaries which are hidden in unsized np arrays.
    # We need to extract them and then re-insert them.
    for key in results.keys():
        if (isinstance(results[key], np.ndarray) 
            and results[key].shape == ()):
            try:
                results[key] = results[key].item()
            except AttributeError:
                pass
        if isinstance(results[key], dict):
            _fix_results_dict(results[key])

def _traverse_and_sort(sub_dict, N, idxs):
    '''Traverse a dictionary and sort any arrays that have the same
    first dimension as the number of targets'''
    for key in sub_dict.keys():
        if isinstance(sub_dict[key], dict):
            _traverse_and_sort(sub_dict[key], N, idxs)
        elif isinstance(sub_dict[key], np.ndarray):
            if sub_dict[key].shape and sub_dict[key].shape[0] == N:
                sub_dict[key] = sub_dict[key][idxs]


def _load_and_sort(results_path):
    results = np.load(results_path, allow_pickle=True)
    results = dict(results) # convert from NPZ file to dict
    _fix_results_dict(results)

    targets = results['targets']
    N = targets.shape[0]
    idxs = np.lexsort((targets[:, -1], targets[:, -2], targets[:, -3]))
    targets = targets[idxs]

    # traverse results dictionary finding all array. 
    # If an array has the same first dimension as the number of targets,
    # sort it according to the idxs
    _traverse_and_sort(results, N, idxs)

    return results


class ComparisonResults:
    def __init__(self, results_path, sort=True):
        self.results_path = results_path
        if sort:
            self.results = _load_and_sort(results_path)
        else:
            results = np.load(results_path, allow_pickle=True)
            results = dict(results) # convert from NPZ file to dict
            _fix_results_dict(results)
            self.results = results

    def get_stim_mat(self, singlespot=True, multispot=True):
        stim_mat = []
        if singlespot:
            stim_mat.append(self.results['singlespot']['stim_mat'])
        if multispot:
            stim_mat.append(self.results['multispot']['stim_mat'])
        return np.concatenate(stim_mat, axis=1)

    def get_singlespot_weights(self, subtracted=True):
        if subtracted:
            return self.results['singlespot']['model_state_subtraction_on']['mu']
        else:
            return self.results['singlespot']['model_state_subtraction_off']['mu']

    def get_multispot_weights(self, subtracted=True):
        if subtracted:
            return self.results['multispot']['model_state_subtraction_on']['mu']
        else:
            return self.results['multispot']['model_state_subtraction_off']['mu']

    def get_singlespot_pscs(self, subtracted=True, neuron_idxs=None):
        stim_mat = self.get_stim_mat(singlespot=True, multispot=False)
        N = stim_mat.shape[0]
        if neuron_idxs is None:
            neuron_idxs = np.arange(N)
        stim_idxs = np.where(stim_mat[neuron_idxs, :].sum(axis=0) > 0)[0]
        if subtracted:
            return self.results['singlespot']['subtracted'][stim_idxs,:]
        else:
            return self.results['singlespot']['raw'][stim_idxs,:]

    def get_multispot_pscs(self, subtracted=True, neuron_idxs=None):
        stim_mat = self.get_stim_mat(singlespot=False, multispot=True)
        N = stim_mat.shape[0]
        if neuron_idxs is None:
            neuron_idxs = np.arange(N)
        stim_idxs = np.where(stim_mat[neuron_idxs, :].sum(axis=0) > 0)[0]
        if subtracted:
            return self.results['multispot']['subtracted'][stim_idxs,:]
        else:
            return self.results['multispot']['raw'][stim_idxs,:]

    def get_singlespot_model_state(self, subtracted=True):
        if subtracted:
            return self.results['singlespot']['model_state_subtraction_on']
        else:
            return self.results['singlespot']['model_state_subtraction_off']

    def get_multispot_model_state(self, subtracted=True):
        if subtracted:
            return self.results['multispot']['model_state_subtraction_on']
        else:
            return self.results['multispot']['model_state_subtraction_off']

    def get_targets(self):
        return self.results['targets']

    def get_singlespot_ests(self):
        return self.results['singlespot']['est']

    def get_multispot_ests(self):
        return self.results['multispot']['est']

    def get_singlespot_demixed(self, subtracted=True):
        if subtracted:
            return self.results['singlespot']['demixed']
        else:
            return self.results['singlespot']['raw_demixed']

    def get_multispot_demixed(self, subtracted=True):
        if subtracted:
            return self.results['multispot']['demixed']
        else:
            return self.results['multispot']['raw_demixed']

    
class GridComparisonResults(ComparisonResults):
    def __init__(self, ss_path=None, ms_path=None, map_idx_start=None, map_idx_end=None):
        ss_results = {}
        ms_results = {}
        if ss_path:
            ss_results = _load_and_sort(ss_path)
            ss_results['singlespot']['powers'] = np.max(ss_results['singlespot']['stim_mat'], axis=0)
            ss_results['singlespot']['targets'] = ss_results['targets']
            del ss_results['targets']
            add_grid_results(ss_results['singlespot'], idx_start=map_idx_start, idx_end=map_idx_end)

        if ms_path:
            ms_results = _load_and_sort(ms_path)
            ms_results['multispot']['powers'] = np.max(ms_results['multispot']['stim_mat'], axis=0)
            ms_results['multispot']['targets'] = ms_results['targets']
            del ms_results['targets']
            add_grid_results(ms_results['multispot'], idx_start=map_idx_start, idx_end=map_idx_end)


        self.results = {**ss_results, **ms_results}

    def get_targets(self):
        pass

    def get_singlespot_targets(self):
        return self.results['singlespot']['targets']

    def get_multispot_targets(self):
        return self.results['multispot']['targets']

    def get_singlespot_weights(self, subtracted=True):
        if subtracted:
            weights = self.results['singlespot']['model_state_subtraction_on']['mu']
        else:
            weights =  self.results['singlespot']['model_state_subtraction_off']['mu']
        return weights.reshape(self.results['singlespot']['raw_map'].shape[1:])

    def get_multispot_weights(self, subtracted=True):
        if subtracted:
            weights = self.results['multispot']['model_state_subtraction_on']['mu']
        else:
            weights =  self.results['multispot']['model_state_subtraction_off']['mu']
        return weights.reshape(self.results['multispot']['raw_map'].shape[1:])