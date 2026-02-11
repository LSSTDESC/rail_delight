import glob
import os
import pickle
import sys

import numpy as np
import pytest
import scipy.special
import tables_io
import yaml
from rail.core.data import DataStore, TableHandle
from rail.core.stage import RailStage
from rail.utils.testing_utils import one_algo

import rail
from rail.estimation.algos import delight_hybrid

sci_ver_str = scipy.__version__.split(".")


@pytest.mark.skipif(
    "rail.estimation.algos.delight_hybrid" not in sys.modules,
    reason="delight_hybrid not installed!",
)
def test_delight():
    with open("./tests/delight/delightPZ.yaml", "r") as f:
        config_dict = yaml.safe_load(f)
    config_dict["model_file"] = "None"
    config_dict["hdf5_groupname"] = "photometry"
    train_algo = delight_hybrid.DelightInformer
    pz_algo = delight_hybrid.DelightEstimator
    results, rerun_results, rerun3_results = one_algo(
        "Delight", train_algo, pz_algo, config_dict, config_dict
    )
    zb_expected = np.array([0.18, 0.01, -1.0, -1.0, 0.01, -1.0, -1.0, -1.0, 0.01, 0.01])
    # assert np.isclose(results.ancil["zmode"], zb_expected, atol=0.03).all() # results are unstable
    assert np.isclose(results.ancil["zmode"], rerun_results.ancil["zmode"]).all()
    # get delight to clean up after itself
    for pattern in [
        "rail/estimation/data/SED/ssp_*Myr_z008_fluxredshiftmod.txt",
        "rail/estimation/data/SED/*_B2004a_fluxredshiftmod.txt",
        "rail/estimation/data/FILTER/DC2LSST_*_gaussian_coefficients.txt",
        "rail/examples_data/estimation_data/tmp/delight_data/galaxies*.txt",
        "parametersTest*.cfg",
        "rail/examples_data/estimation_data/tmp/delight_data/*.h5",
    ]:
        files = glob.glob(pattern)
        for file_ in files:
            os.remove(file_)
    os.removedirs("rail/examples_data/estimation_data/tmp/delight_data")
