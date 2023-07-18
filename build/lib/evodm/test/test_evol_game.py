from evodm import evol_env, generate_landscapes, normalize_landscapes, run_sim
from evodm.evol_game import discretize_state, define_mira_landscapes, evol_env_wf
import pytest
import numpy.testing as npt
import numpy as np
import pandas as pd
import random

#start testing that the environment class is working as expected
@pytest.fixture
def env_init():
    return evol_env(normalize_drugs=True, random_start = False, num_evols =3, add_noise = False)

@pytest.fixture
def env_mature():
    env = evol_env(normalize_drugs=True, random_start = False, num_evols =3, add_noise = False)
    for i in range(100):
        env.action = random.randint(np.min(env.ACTIONS),np.max(env.ACTIONS))
        env.step()
    return env

@pytest.fixture
def env_total_resistance():
    env = evol_env(normalize_drugs=True, random_start = False, num_evols =1, 
                   add_noise = False, total_resistance= True)
    for i in range(100):
        env.action = random.randint(np.min(env.ACTIONS),np.max(env.ACTIONS))
        env.step()
    return env

@pytest.fixture
def popsize_env():
    test_env = evol_env(train_input = "pop_size", add_noise = False)
    for i in range(3):
        test_env.step()
    return test_env 

@pytest.fixture
def env_small():
    env_small = evol_env(normalize_drugs=True, N=3, random_start = False, 
                         num_evols =1, add_noise = False)
    return env_small
    #make a list of second states

@pytest.fixture
def env_noise():
    env = evol_env(normalize_drugs=True, random_start = False, num_evols =1,
                   add_noise = True, noise_modifier=5, train_input="fitness", 
                   win_threshold=10000)
    for i in range(100):
        env.action = random.randint(np.min(env.ACTIONS),np.max(env.ACTIONS))
        env.step()
    env_noise = env
    return env_noise

#tests to make sure noise is being encoded properly
def test_noise1(env_noise):
    assert env_noise.sensor_fitness != env_noise.fitness

def test_noise2(env_noise):
    assert env_noise.sensor[2] == 1-env_noise.fitness


#function that ensures population is only transitioning by single mutations along the fitness landscape
def test_traversal(env_small):
    env=env_small
    second_states = []
    for i in range(1000):
        #ppick a random action
        env.action = random.randint(np.min(env.ACTIONS),np.max(env.ACTIONS))
        env.step()
        second_states.append(np.argmax(env.state_vector))
        env.state_vector  = np.zeros((2**3,1))
        env.state_vector[0][0] = 1

    assert np.all([i in [0,1,2,4] for i in second_states])

def test_traversal2(env_small):
    env=env_small
    second_states = []
    env.state_vector  = np.zeros((2**3,1))
    env.state_vector[1][0] = 1
    for i in range(1000):
        #ppick a random action
        env.action = random.randint(np.min(env.ACTIONS),np.max(env.ACTIONS))
        env.step()
        second_states.append(np.argmax(env.state_vector))
        env.state_vector  = np.zeros((2**3,1))
        env.state_vector[1][0] = 1

    assert np.all([i in [0,1,3,5] for i in second_states])

def test_traversal3(env_small):
    env=env_small
    second_states = []
    env.state_vector  = np.zeros((2**3,1))
    env.state_vector[7][0] = 1
    for i in range(1000):
        #ppick a random action
        env.action = random.randint(np.min(env.ACTIONS),np.max(env.ACTIONS))
        env.step()
        second_states.append(np.argmax(env.state_vector))
        env.state_vector  = np.zeros((2**3,1))
        env.state_vector[7][0] = 1

    assert np.all([i in [7,3,5,6] for i in second_states])

def test_traversal4(env_small):
    env=env_small
    second_states = []
    env.state_vector  = np.zeros((2**3,1))
    env.state_vector[7][0] = 1
    for i in range(1000):
        #ppick a random action
        env.action = random.randint(np.min(env.ACTIONS),np.max(env.ACTIONS))
        env.step()
        second_states.append(np.argmax(env.state_vector))
        env.state_vector  = np.zeros((2**3,1))
        env.state_vector[7][0] = 1

    assert len(pd.unique(second_states)) <= 4

def test_traversal5(env_small):
    env=env_small
    second_states = []
    env.state_vector  = np.zeros((2**3,1))
    env.state_vector[3][0] = 1
    for i in range(1000):
        #ppick a random action
        env.action = random.randint(np.min(env.ACTIONS),np.max(env.ACTIONS))
        env.step()
        second_states.append(np.argmax(env.state_vector))
        env.state_vector  = np.zeros((2**3,1))
        env.state_vector[3][0] = 1

    assert np.all([i in (1,2,3,7) for i in second_states])

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
    assert test_env.ENVIRONMENT_SHAPE[0] == 5

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
    if len(fitness) == 1:
        checkfitness = 0 <= fitness <= 1
    else:
        checkfitness = all([0 <= i <= 1 for i in fitness])
    checkstate = all([0 <= i <= 1 for i in state_vector])
    assert all([checkfitness, checkstate])

#check that we can adequately turn a population with many possible futures into a single population
def test_discretize_state1():
    state_vector = np.array([0,0.2,0.2,0, 0, 0, 0, 0.6])
    states = state_vector.reshape((len(state_vector), 1))
    new_states = discretize_state(states)
    assert np.max(new_states) == 1

#do it again - this time be sure it doesn't break if you give it something that already has the population all in one state
def test_discretize_state2():
    state_vector = np.array([0,0,1,0, 0, 0, 0, 0])
    states = state_vector.reshape((len(state_vector), 1))
    new_states = discretize_state(states)
    assert all([state_vector[i] == new_states[i] for i in range(len(state_vector))])

def test_discretize_state3():
    state_vector = np.array([0,0,0.2,0, 0, 0.4, 0, 0.4])
    states = state_vector.reshape((len(state_vector), 1))
    new_states = discretize_state(states)
    assert np.argmax(new_states) in [2,5,7]

#make sure it can go without averaging the evolutionary outcomes.
def test_run_sim2(env_mature):
    fitness, state_vector = run_sim(evol_steps = env_mature.NUM_EVOLS, N = env_mature.N,
                                           sigma = env_mature.sigma,
                                           state_vector = env_mature.state_vector,
                                           drugs = env_mature.drugs, action = env_mature.action, 
                                           average_outcomes=False)
    if len(fitness) == 1:
        checkfitness = 0 <= fitness <= 1
    else:
        checkfitness = all([0 <= i <= 1 for i in fitness])
    checkstate = all([i == 1 or i == 0 for i in state_vector])
    assert all([checkfitness, checkstate])
#just make sure Jeff's code doesn't break while we're doing other things
@pytest.fixture
def example_landscapes():

    landscapes, drugs = generate_landscapes(N=5, dense=True,num_drugs = 4)
    return [landscapes,drugs]

def test_generate_landscapes(example_landscapes):
    assert len(example_landscapes[0]) ==4
    assert len(example_landscapes[1][0]) == pow(2,5)


def test_normalize_landscapes(example_landscapes):
    drugs = example_landscapes[1]
    drugs = normalize_landscapes(drugs)
    maxes = [np.max(drug) <= 1 for drug in drugs]
    mins = [np.min(drug) >= 0 for drug in drugs] 
    assert all([mins, maxes])


#need a test to make sure we can compute total resistance across a panel. 
def test_total_resistance(env_total_resistance):
    assert isinstance(env_total_resistance.fitness - 1, np.float)

@pytest.fixture
def mira_env():
    drugs = define_mira_landscapes()
    mira_env = evol_env(N=4, drugs = drugs, num_drugs = 15, normalize_drugs=False,
                   train_input = 'fitness')
    for i in range(100):
        mira_env.action = random.randint(np.min(mira_env.ACTIONS),np.max(mira_env.ACTIONS))
        mira_env.step()
    mira_env.action = 5 #6 instead of 5 because in my infinite wisdom I used different indexing systems for these
    mira_env.step()
    mira_env.step()
    return mira_env

@pytest.fixture
def mira_env_wf(): 
    mira_env_wf = evol_env_wf()
    for i in range(100):
        mira_env_wf.action = random.randint(np.min(mira_env_wf.ACTIONS),np.max(mira_env_wf.ACTIONS))
        mira_env_wf.step()
    
    mira_env_wf.action = 5
    mira_env_wf.step()
    mira_env_wf.step()
    return mira_env_wf

def test_convert_fitness_wf(mira_env_wf, mira_env):
    assert len(mira_env_wf.sensor[0]) == len(mira_env.sensor[0])

def test_init_sparse():
    env = evol_env(dense=False)
    ls1 = env.landscapes[0]
    assert ls1.get_TM().size < 4**env.N

def test_init_dense():
    env = evol_env(dense=True)
    ls1 = env.landscapes[0]
    assert ls1.get_TM().size == 4**env.N






    