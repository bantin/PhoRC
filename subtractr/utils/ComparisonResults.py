import numpy as np

class ComparisonResults:
    def __init__(self, results_path):
        results = np.load(results_path, allow_pickle=True)
        self.results = results

    def get_stim_mat(self, singlespot=True, multispot=True):
        stim_mat = []
        if singlespot:
            stim_mat.append(self.results['singlespot'].item()['stim_mat'])
        if multispot:
            stim_mat.append(self.results['multispot'].item()['stim_mat'])
        return np.concatenate(stim_mat, axis=1)

    def get_singlespot_weights(self, subtracted=True):
        if subtracted:
            return self.results['singlespot'].item()['model_state_subtraction_on']['mu']
        else:
            return self.results['singlespot'].item()['model_state_subtraction_off']['mu']

    def get_multispot_weights(self, subtracted=True):
        if subtracted:
            return self.results['multispot'].item()['model_state_subtraction_on']['mu']
        else:
            return self.results['multispot'].item()['model_state_subtraction_off']['mu']

    def get_singlespot_pscs(self, subtracted=True, neuron_idxs=None):
        stim_mat = self.get_stim_mat(singlespot=True, multispot=False)
        N = stim_mat.shape[0]
        if neuron_idxs is None:
            neuron_idxs = np.arange(N)
        stim_idxs = np.where(stim_mat[neuron_idxs, :].sum(axis=0) > 0)[0]
        if subtracted:
            return self.results['singlespot'].item()['subtracted'][stim_idxs,:]
        else:
            return self.results['singlespot'].item()['raw'][stim_idxs,:]

    def get_singlespot_model_state(self, subtracted=True):
        if subtracted:
            return self.results['singlespot'].item()['model_state_subtraction_on']
        else:
            return self.results['singlespot'].item()['model_state_subtraction_off']

    def get_targets(self):
        return self.results['targets']

    def get_singlespot_ests(self):
        return self.results['singlespot'].item()['est']

    def get_singlespot_demixed(self, subtracted=True):
        if subtracted:
            return self.results['singlespot'].item()['demixed']
        else:
            return self.results['singlespot'].item()['raw_demixed']