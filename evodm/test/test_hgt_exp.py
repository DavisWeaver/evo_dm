import pytest
from evodm.landscapes import *
from evodm.hgt_exp import *

def test_run_sim_hgt():
    ls = Landscape(N=5, sigma=0.5, num_jumps = 2)