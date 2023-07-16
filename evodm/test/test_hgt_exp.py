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

@pytest.fixture
def wf_smallpop():
    wf_smallpop = summarize_wf_hgt(pop_size = 10000, theta = 1e-5, stop_count = 100)
    return wf_smallpop

#nake sure pop fixes at single allele in small pop limit 
#def test_wf_sp_fixation(wf_smallpop):
    df = wf_smallpop[0]
    fixation = df[df.step_num == np.max(df.step_num)]
    assert fixation['num_alleles'].tolist()[0] ==1

#make sure evolution is actually happening
def test_wf_sp_fixation2(wf_smallpop):
    df = wf_smallpop[0]
    fixation = df[df.step_num == np.max(df.step_num)]
    assert fixation['dom_allele'].tolist()[0] != '00000'

#Check out the large pop limit
@pytest.fixture
def wf_largepop():
    wf_largepop= summarize_wf_hgt(pop_size=1e6, theta=1e-5, stop_count=200)
    return wf_largepop

def test_wf_lp_fixation(wf_largepop):
    df= wf_largepop[0]
    fixation = df[df.step_num == np.max(df.step_num)]
    assert fixation['num_alleles'].tolist()[0] > 1 #make sure we are actually in large pop regime

def test_wf_lp_fixation2(wf_largepop):
    df= wf_largepop[0]
    fixation = df[df.step_num == np.max(df.step_num)]
    assert fixation['fitness'].tolist()[0] > 0.9999 #large pop should be reaching global maximum

@pytest.fixture
def wf_lp_N4():
    wp_lp_N4 = summarize_wf_hgt(pop_size=1e6, theta=1e-5, stop_count = 200, N=4)
    return wp_lp_N4

def test_wf_lp_fixation3(wf_lp_N4):
    df= wf_lp_N4[0]
    fixation = df[df.step_num == np.max(df.step_num)]
    assert fixation['step_num'].tolist()[0] < 10000 #make sure appropriately stop when fixation is achieved


def test_wf_lp_fixation4():
    wp_lp_N6 = summarize_wf_hgt(pop_size=1e6, theta=1e-5, stop_count = 200, N=6)
    df= wp_lp_N6[0]
    fixation = df[df.step_num == np.max(df.step_num)]
    assert fixation['step_num'].tolist()[0] < 10000 #make sure appropriately stop when fixation is achieved

def test_wf_lp_fixation5():
    wp_lp = summarize_wf_hgt(pop_size=1e6, theta=1e-5, stop_count = 200, N=7)
    df= wp_lp[0]
    fixation = df[df.step_num == np.max(df.step_num)]
    assert fixation['step_num'].tolist()[0] < 10000 #make sure appropriately stop when fixation is achieved

def test_wf_lp_fixation6():
    wp_lp = summarize_wf_hgt(pop_size=1e6, theta=1e-5, stop_count = 200, N=8)
    df= wp_lp[0]
    fixation = df[df.step_num == np.max(df.step_num)]
    assert fixation['step_num'].tolist()[0] < 10000 #make sure appropriately stop when fixation is achieved