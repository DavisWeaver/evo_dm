from evodm import evol_env, generate_landscapes, define_drugs, normalize_landscapes, run_sim 
import pytest


def test_init_state_envshape(): 
    test_env = evol_env(train_input = 'state_vector')
    #Test that the environment shape is equal to the length of the state vector
    assert test_env.ENVIRONMENT_SHAPE[0] == 32

def test_init_fitness_envshape():
    test_env = evol_env(train_input = "fitness", num_evols = 1)
    #test that the environment shape is equal to the length of the number of evols
    assert test_env.ENVIRONMENT_SHAPE[0] == 1

def test_init_popsize_envshape():
    test_env = evol_env(train_input = "pop_size")
    #test that the environment shape is equal to 100, the number of "OD measurements" we are simulating
    assert test_env.ENVIRONMENT_SHAPE[0] == 100

@pytest.fixture
def popsize_env():
    test_env = evol_env(train_input = "pop_size")
    for i in range(2):
        test_env.step()
    return test_env 

def test_step_popsize(popsize_env):
    #test
    assert popsize_env.sensor[0][99] == popsize_env.sensor[3][0]

