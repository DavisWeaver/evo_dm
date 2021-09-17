from evodm import evol_env, generate_landscapes, define_drugs, normalize_landscapes, run_sim
import pytest
import numpy.testing as npt
import numpy as np

#start testing that the environment class is working as expected
@pytest.fixture
def env_init():
    return evol_env(normalize_drugs=True, random_start = False)

@pytest.fixture
def popsize_env():
    test_env = evol_env(train_input = "pop_size")
    for i in range(2):
        test_env.step()
    return test_env 

#Test that the evol_env environment is being initialized properly under a variety of conditions
def test_init_state(env_init): 
    assert env_init.state_vector[0][0] == 1

def test_init_normalized(env_init): 
    maxes = [np.max(drug) <= 1 for drug in env_init.drugs]
    mins = [np.min(drug) >= 0 for drug in env_init.drugs] 
    assert all([mins, maxes])

def test_init_fitness(env_init):
    assert env_init.fitness == env_init.drugs[0][0]

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

def test_init_randomstart(): 
    test_env = evol_env(random_start = True)
    bool_list = [i == test_env.state_vector[0] for i in test_env.state_vector]
    assert all(bool_list)

def test_step_popsize(popsize_env):
    #test that the first step 
    assert popsize_env.sensor[0][99] == popsize_env.sensor[3][0]

def test_popsize_behavior(popsize_env):
    npt.assert_almost_equal(popsize_env.sensor[3][99], popsize_env.fitness, decimal = 3)

def test_env_reset(popsize_env):
    popsize_env.reset()
    assert popsize_env.fitness == popsize_env.drugs[0][0]

#lets test the standalone functions
def test_run_sim(env_init):
    fitness, state_vector = run_sim(evol_steps = env_init.NUM_EVOLS, N = env_init.N,
                                           sigma = env_init.sigma,
                                           state_vector = env_init.state_vector,
                                           drugs = env_init.drugs, action = env_init.action)
    checkfitness = 0 <= fitness <= 1
    checkstate = all([0 <= i <= 1 for i in state_vector])
    assert all([checkfitness, checkstate])

#just make sure Jeff's code doesn't break while we're doing other things
@pytest.fixture
def example_landscapes():
    return generate_landscapes(N=5)

def test_generate_landscapes(example_landscapes):
    assert len(example_landscapes) > 25

def test_define_drugs(example_landscapes):
    #just check that there are 4 drugs vecs of length 2^5
    drugs = define_drugs(example_landscapes, num_drugs = 4)
    four_drugs = len(drugs) == 4
    five_N = len(drugs[0]) == pow(2,5)
    assert all([four_drugs, five_N])

def test_normalize_landscapes(example_landscapes):
    drugs = define_drugs(example_landscapes, num_drugs = 4)
    drugs = normalize_landscapes(drugs)
    maxes = [np.max(drug) <= 1 for drug in env_init.drugs]
    mins = [np.min(drug) >= 0 for drug in env_init.drugs] 
    assert all([mins, maxes])
    