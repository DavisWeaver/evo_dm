import pytest
from evodm.landscapes import *
from evodm.hgt_exp import *
from evodm.data import *

def test_run_sim_hgt():
    ls = Landscape(N=5, sigma=0.5, num_jumps = 2)

def test_data_grab():
    drug = get_example_drug(N=5)
    assert len(drug) == 32

def test_data_grab2():
    drug = get_example_drug(N=6)
    assert len(drug) == 64

def test_data_grab3():
    drug = get_example_drug(N=7)
    assert len(drug) == 128

def test_data_grab4():
    drug = get_example_drug(N=8)
    assert len(drug) == 256

def test_data_grab5():
    drug = get_example_drug(N=4)
    assert len(drug) == 16
