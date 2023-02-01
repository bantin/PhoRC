import argparse

def add_subtraction_args(parser=None):
    """
    Add arguments for the subtraction method to an argument parser.

    Parameters
    ----------
    argseq : list
        List of arguments to add to the parser. Each element of the list should be
        a tuple of the form (name, type, default, help). The type should be a
        callable that can be used to convert the argument to the desired type.
    parser : argparse.ArgumentParser, optional
        The parser to add the arguments to. If None, a new parser is created.

    Returns
    -------
    parser : argparse.ArgumentParser
        The parser with the arguments added.

    Note: Adds the following arguments:
        --subtraction_method : str
            The method to use for subtracting the photocurrent. Can be 'network' or
            'low_rank'. If 'network', the network will be used to estimate the
            photocurrent. If 'low_rank', the low-rank subtraction method will be
            used.
        --stim_start_idx : int
            The index of the stimulus start to use for the subtraction. Only used
            if subtraction_method is 'low_rank'.
        --stim_end_idx : int
            The index of the last stimulus end to use for the subtraction. Only used
            if subtraction_method is 'low_rank'.
        --rank : int
            The rank of the low-rank subtraction. Only used if subtraction_method is
            'low_rank'.
        --constrain_V : bool
            Whether to constrain the V matrix in the low-rank subtraction. Only used
            if subtraction_method is 'low_rank'.
        --batch_size : int
            The batch size to use for the subtraction.
        --baseline : bool
            Whether to fit a baseline term.
            Only used if subtraction_method is 'low_rank'.
        --subtract_baseline : bool
            Whether to subtract the baseline from the traces after subtracting the
            photocurrent. Only used if subtraction_method is 'low_rank'.
            
    """
    if parser is None:
        parser = argparse.ArgumentParser()
    
    # whether we should subtract at all
    parser.add_argument('--subtract_pc', action='store_true')
    parser.add_argument('--no_subtract_pc', dest='subtract_pc', action='store_false')
    parser.set_defaults(subtract_pc=True)

    # arguments used for the network subtraction, 
    # can be either 'network' or 'low_rank'
    parser.add_argument('--subtraction_method', type=str, default='low_rank')
    parser.add_argument('--subtractr_path', type=str, default=None)

    # arguments used for the low-rank subtraction, if the subtraction_method is
    # 'network' these arguments are ignored
    parser.add_argument('--stim_start_idx', type=int, default=100)
    parser.add_argument('--stim_end_idx', type=int, default=200)
    parser.add_argument('--rank', type=int, default=2)
    parser.add_argument('--constrain_V', action='store_true')
    parser.add_argument('--no_constrain_V', dest='constrain_V', action='store_false')
    parser.set_defaults(constrain_V=True)
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--baseline', action='store_true')
    parser.add_argument('--no_baseline', dest='baseline', action='store_false')
    parser.set_defaults(baseline=True)
    parser.add_argument('--subtract_baseline', action='store_true')
    parser.add_argument('--no_subtract_baseline', dest='subtract_baseline', action='store_false')
    parser.set_defaults(baseline=True)

    return parser

def add_caviar_args(parser=None):
    """
    Parse arguments for running caviar

    Parameters
    ----------
    argseq : list
        list of arguments to parse
    parser : argparse.ArgumentParser, optional
        parser to add arguments to. If None, a new parser is created

    Adds the following arguments:
    --run-caviar : bool, optional
        if True, run caviar on the data.
    --minimum-spike-count : int, optional
        minimum number of spikes to consider a neuron connected
    --msrmp : float, optional
        minimum spike rate at max power
    --iters : int, optional
        number of iterations to run caviar
    --save-histories : bool, optional
        if True, save the history of the caviar run

    """
    if parser is None:
        parser = argparse.ArgumentParser()
    parser.add_argument('--run_caviar', action='store_true', default=False)
    parser.add_argument('--minimum_spike_count', type=int, default=3)
    parser.add_argument('--msrmp', type=float, default=0.3)
    parser.add_argument('--iters', type=int, default=30)
    parser.add_argument('--save-histories', action='store_true', default=False)
    
    return parser