import numpy as np
from subtractr.utils.grid_utils import sort_results

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

class ComparisonResults:
    def __init__(self, results_path, sort=True):
        results = np.load(results_path, allow_pickle=True)
        self.results = dict(results) # convert from NPZ file to dict
        _fix_results_dict(self.results)

        if sort:

            def _traverse_and_sort(sub_dict):
                '''Traverse a dictionary and sort any arrays that have the same
                first dimension as the number of targets'''
                # import pdb; pdb.set_trace()
                for key in sub_dict.keys():
                    if isinstance(sub_dict[key], dict):
                        _traverse_and_sort(sub_dict[key])
                    elif isinstance(sub_dict[key], np.ndarray):
                        if sub_dict[key].shape and sub_dict[key].shape[0] == N:
                            print('found array to sort: %s' % key)
                            sub_dict[key] = sub_dict[key][idxs]

            targets = results['targets']
            N = targets.shape[0]
            idxs = np.lexsort((targets[:, -1], targets[:, -2], targets[:, -3]))
            targets = targets[idxs]
        
            # traverse results dictionary finding all array. 
            # If an array has the same first dimension as the number of targets,
            # sort it according to the idxs
            _traverse_and_sort(self.results)


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

    def get_singlespot_model_state(self, subtracted=True):
        if subtracted:
            return self.results['singlespot']['model_state_subtraction_on']
        else:
            return self.results['singlespot']['model_state_subtraction_off']

    def get_targets(self):
        return self.results['targets']

    def get_singlespot_ests(self):
        return self.results['singlespot']['est']

    def get_singlespot_demixed(self, subtracted=True):
        if subtracted:
            return self.results['singlespot']['demixed']
        else:
            return self.results['singlespot']['raw_demixed']