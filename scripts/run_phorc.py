import hydra
from omegaconf import DictConfig, OmegaConf
import phorc
import numpy as np
import h5py
import os
import shutil
import json


@hydra.main(version_base=None, config_path="conf", config_name="run_phorc_config")
def main(cfg: DictConfig) -> None:

    dset_name = os.path.basename(cfg.dataset_path).split('.')[0]

    # Load current data
    with h5py.File(cfg.dataset_path) as f:
        pscs = np.array(f['pscs']).T

    # Run phorc using **subtract_args
    # Convert the subtraction part of the configuration to a plain dictionary
    phorc_args = cfg["phorc"]
    estimate_args = cfg["estimate"]

    est = phorc.estimate(pscs, **estimate_args, **phorc_args)
    pscs_subtracted = pscs - est

    # Create string containing arguments used when running PhoRC
    argstr = json.dumps(OmegaConf.to_container(cfg, resolve=True))
    print(argstr)

    # Save the results
    outfile_name = os.path.join(cfg.save_path, dset_name + '_phorc.h5')

    # Copy the input file to the output file, then add the results to the output file
    shutil.copyfile(cfg.dataset_path, outfile_name)

    with h5py.File(outfile_name, 'a') as outfile:
        if "pscs_subtracted" in outfile:
            del outfile["pscs_subtracted"]
        outfile.create_dataset("pscs_subtracted", data=pscs_subtracted)
        outfile["pscs_subtracted"].attrs["args"] = argstr


if __name__ == "__main__":
    main()
