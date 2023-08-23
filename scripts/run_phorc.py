import numpy as np
import h5py
import sys
import json
import phorc
import phorc.utils as utils
import os
import argparse
import shutil


def parse_fit_options(argseq):
    parser = argparse.ArgumentParser(
        description='Opsin subtraction + CAVIaR for Grid Denoising')

    parser = utils.add_subtraction_args(parser=parser)
    parser.add_argument('--dataset_path', type=str)
    parser.add_argument('--save_path', type=str)
    args = parser.parse_args(argseq)

    return args


if __name__ == "__main__":
    args = parse_fit_options(sys.argv[1:])
    dset_name = os.path.basename(args.dataset_path).split('.')[0]

    # load current data
    with h5py.File(args.dataset_path) as f:
        pscs = np.array(f['pscs']).T

    # run phorc
    est = phorc.estimate(pscs,
        rank=args.rank,
        batch_size=args.batch_size,
        window_start=args.window_start_idx,
        window_end=args.window_end_idx,
        const_baseline=args.const_baseline,
        decaying_baseline=args.decaying_baseline)
    pscs_subtracted = pscs - est

    # Create string containing arguments
    # used when running PhoRC
    argstr = json.dumps(args.__dict__)

    # save the results
    outfile_name = os.path.join(args.save_path, dset_name + '_phorc.h5')

    # copy the input file to the output file,
    # then add the results to the output file
    shutil.copyfile(args.dataset_path, outfile_name)

    with h5py.File(outfile_name, 'a') as outfile:
        if "pscs_subtracted" in outfile:
            del outfile["pscs_subtracted"]
        outfile.create_dataset("pscs_subtracted", data=pscs_subtracted)
        outfile["pscs_subtracted"].attrs["args"] = argstr



