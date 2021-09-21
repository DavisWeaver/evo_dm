from evodm import DrugSelector, practice, hyperparameters
import random
import numpy as np
import pytest

#define fixtures to use for testing functions - lots of these will depend on each other
@pytest.fixture
def ds():
    hp = hyperparameters()
    hp.TRAIN_INPUT = "fitness"
    hp.MIN_REPLAY_MEMORY_SIZE = 20
    hp.MINIBATCH_SIZE = 10
    hp.RESET_EVERY = 20
    hp.EPISODES = 10
    hp.N = 4
    ds = DrugSelector(hp = hp)
    return ds

@pytest.fixture
def ds_replay(ds):
    for episode in range(1, ds.hp.EPISODES + 1):
        for i in range(1, ds.hp.RESET_EVERY):
            ds.env.action = random.randint(np.min(ds.env.ACTIONS),np.max(ds.env.ACTIONS))
            ds.env.step()
            ds.update_replay_memory()
        ds.env.reset()
    return ds

@pytest.fixture
def minibatch(ds_replay):
    minibatch = random.sample(ds_replay.replay_memory, ds_replay.hp.MINIBATCH_SIZE)
    return minibatch

#test get current states --> process by which we grab the current state and next state from the replay memory s
@pytest.fixture
def current_states(ds_replay, minibatch):
    current_states, new_current_states = ds_replay.get_current_states(minibatch = minibatch)
    return current_states

@pytest.fixture
def new_current_states(ds_replay, minibatch):
    current_states, new_current_states = ds_replay.get_current_states(minibatch = minibatch)
    return new_current_states

#need input of shape 2,
def test_current_states_shape(current_states, ds):
    assert current_states[0].shape == ds.env.ENVIRONMENT_SHAPE

#make sure all the actions fall in the defined action space
def test_current_states_actions(current_states, ds):
    action_bools = [i[0] in ds.env.ACTIONS for i in current_states]
    assert all(action_bools)

def test_current_states_fitness(current_states):
    fitness_bools = [i[1] >= 0 and i[1] <= 1 for i in current_states]
    assert all(fitness_bools)

def test_new_current_states_actions(new_current_states, ds):
    action_bools = [i[0] in ds.env.ACTIONS for i in new_current_states]
    assert all(action_bools)

def test_new_current_states_fitness(new_current_states):
    fitness_bools = [i[1] >= 0 and i[1] <= 1 for i in new_current_states]
    assert all(fitness_bools)

#test the enumerate batch function
@pytest.fixture
def batch_enumerated(ds_replay, current_states, new_current_states, minibatch):
    current_qs_list = ds_replay.model.predict(current_states)
    future_qs_list = ds_replay.model.predict(new_current_states)
    x,y = ds_replay.enumerate_batch(minibatch = minibatch, future_qs_list = future_qs_list, 
                                    current_qs_list = current_qs_list)


def test_add_noise(ds):
    #step forward so there is actually something in the sensor
    ds.env.step()
    
    #check to see if the add_noise functionality is actually scrambling things up.
    bool_list = []
    for i in range(3):
        ds.env.step()
        fitness = ds.env.sensor[0][1]
        ds.add_noise()
        new_fitness = ds.env.sensor[0][1]
        bool_list.append(new_fitness != fitness)
    
    assert any(bool_list)


