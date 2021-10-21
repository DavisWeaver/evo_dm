from evodm import DrugSelector, practice, hyperparameters
import random
import numpy as np
import pytest
from itertools import chain

#define fixtures to use for testing functions - lots of these will depend on each other
@pytest.fixture
def hp():
    hp = hyperparameters()
    hp.TRAIN_INPUT = "fitness"
    hp.MIN_REPLAY_MEMORY_SIZE = 20
    hp.MINIBATCH_SIZE = 10
    hp.RESET_EVERY = 20
    hp.EPISODES = 10
    hp.N = 4
    return hp

@pytest.fixture
def hp_state():
    hp_state = hyperparameters()
    hp_state.TRAIN_INPUT = "state_vector"
    hp_state.MIN_REPLAY_MEMORY_SIZE = 20
    hp_state.MINIBATCH_SIZE = 10
    hp_state.RESET_EVERY = 20
    hp_state.EPISODES = 10
    hp_state.N = 4
    return hp_state

@pytest.fixture
def ds(hp):
    ds = DrugSelector(hp = hp)
    return ds

@pytest.fixture
def ds_state(hp_state):
    ds_state = DrugSelector(hp=hp_state)
    for episode in range(1, ds_state.hp.EPISODES + 1):
        for i in range(1, ds_state.hp.RESET_EVERY):
            ds_state.env.action = random.randint(np.min(ds_state.env.ACTIONS),np.max(ds_state.env.ACTIONS))
            ds_state.env.step()
            ds_state.update_replay_memory()
        ds_state.env.reset()
    return ds_state


@pytest.fixture
def ds_replay(hp):
    ds_replay = DrugSelector(hp = hp)
    for episode in range(1, ds_replay.hp.EPISODES + 1):
        for i in range(1, ds_replay.hp.RESET_EVERY):
            ds_replay.env.action = random.randint(np.min(ds_replay.env.ACTIONS),np.max(ds_replay.env.ACTIONS))
            ds_replay.env.step()
            ds_replay.update_replay_memory()
        ds_replay.env.reset()
    return ds_replay


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

#test the enumerate batch function
@pytest.fixture
def batch_enumerated(ds_replay, current_states, new_current_states, minibatch):
    current_qs_list = ds_replay.model.predict(current_states)
    future_qs_list = ds_replay.model.predict(new_current_states)
    x,y = ds_replay.enumerate_batch(minibatch = minibatch, future_qs_list = future_qs_list, 
                                    current_qs_list = current_qs_list)
    return [x,y]

#don't need to test the intermediate steps because they are the same regardless of train input
@pytest.fixture
def current_states_state(ds_state):
    minibatch = random.sample(ds_state.replay_memory, ds_state.hp.MINIBATCH_SIZE)
    current_states, new_current_states = ds_state.get_current_states(minibatch = minibatch)
    return [current_states, new_current_states]

#need input of shape 2,
def test_current_states_shape(current_states, ds):
    assert current_states[0].shape == ds.env.ENVIRONMENT_SHAPE

#make sure all the actions fall in the defined action space
def test_current_states_actions(current_states, ds):

    actions_one_hot = [i[:len(ds.env.ACTIONS)] for i in current_states]
    #convert the one hot action back into categorical and verify that they are all in ds.env.actions
    action_bools = [np.dot(i, ds.env.ACTIONS) in ds.env.ACTIONS for i in actions_one_hot]
    assert all(action_bools)

def test_current_states_fitness(current_states, ds):
    fitness_bools = [i[len(ds.env.ACTIONS):] >= 0 and i[len(ds.env.ACTIONS):] <= 1 for i in current_states]
    assert all(fitness_bools)

def test_new_current_states_actions(new_current_states, ds):
    actions_one_hot = [i[:len(ds.env.ACTIONS)] for i in new_current_states]
    #convert the one hot action back into categorical and verify that they are all in ds.env.actions
    action_bools = [np.dot(i, ds.env.ACTIONS) in ds.env.ACTIONS for i in actions_one_hot]
    assert all(action_bools)

def test_new_current_states_fitness(new_current_states, ds):
    fitness_bools = [i[len(ds.env.ACTIONS):] >= 0 and i[len(ds.env.ACTIONS):] <= 1 for i in new_current_states]
    assert all(fitness_bools)

#This section of the code used to break when you use state_vector as the train_input
def test_predict_qs_state(current_states_state, ds_state):
    current_states = current_states_state[0]
    current_qs_list = ds_state.model.predict(current_states)
    assert len(current_states) == len(current_qs_list)

#This section of the code used to break when you use state_vector as the train_input
def test_predict_qs_state2(current_states_state, ds_state):
    current_states = current_states_state[0]
    current_qs_list = ds_state.model.predict(current_states)
    assert len(current_qs_list[0]) == len(ds_state.env.ACTIONS)

#now verify that all q values are valid floats
def test_predict_qs_state3(current_states_state, ds_state):
    #setup
    current_states = current_states_state[0]
    current_qs_list = ds_state.model.predict(current_states)
    
    #this line flattens the list
    qs = list(chain.from_iterable(current_qs_list))
    bools = [isinstance(i, np.floating) for i in qs] #np type floats don't work with regular python type checking - super strange

    assert all(bools)

def test_get_qs(ds_replay):
    qs = ds_replay.get_qs()
    assert len(qs) == len(ds_replay.env.ACTIONS)

#test it for ds_state as well
def test_get_qs_state(ds_state):
    qs = ds_state.get_qs()
    assert len(qs) == len(ds_state.env.ACTIONS)

def test_get_qs2(ds_replay,ds):
    for i in range(3):
        ds.env.step()
        ds.update_replay_memory()
    
    #just make sure q changes based on the inputs
    qs1 = ds.get_qs()
    qs2 = ds_replay.get_qs()

    bools = [qs1[i] != qs2[i] for i in range(len(qs1))]
    assert any(bools)

def test_get_qs2_state(ds_state):
    #just make sure q changes based on the inputs
    qs1 = ds_state.get_qs()
    for i in range(20):
        ds_state.env.action = random.randint(np.min(ds_state.env.ACTIONS),np.max(ds_state.env.ACTIONS))
        ds_state.env.step()
        ds_state.update_replay_memory()
    qs2 = ds_state.get_qs()
    for i in range(20):
        ds_state.env.action = random.randint(np.min(ds_state.env.ACTIONS),np.max(ds_state.env.ACTIONS))
        ds_state.env.step()
        ds_state.update_replay_memory()
    qs3 = ds_state.get_qs()

    bools = [qs1[i] != qs2[i] for i in range(len(qs1))]
    bools2 = [qs1[i] != qs3[i] for i in range(len(qs1))]
    assert any([bools, bools2])
    
def test_enumerate_batch_x(batch_enumerated):
    x = batch_enumerated[0]
    assert x.shape == (10,5) #10 is the minibatch size that we set in the hp fixture, 5 because 4 drugs + one fitness value
def test_enumerate_batch_y(batch_enumerated):
    y = batch_enumerated[1]
    assert y.shape == (10,4) #4 because there are 4 q-values. One for each drug

    #Test that we are actually changing the weights when we train
def test_update_weights(ds_replay, batch_enumerated):
    X = batch_enumerated[0]
    y = batch_enumerated[1]
    weights_1 = ds_replay.model.get_weights()[1]
    ds_replay.model.fit(X, y, batch_size=ds_replay.hp.MINIBATCH_SIZE, 
                       verbose=0, shuffle=False, callbacks=None)
    weights_2 = ds_replay.model.get_weights()[1]
    bools = [weights_1[i] != weights_2[i] for i in range(len(weights_1))]
    #test that the weights are being updated
    assert any(bools)

#def test_compute_optimal_action(ds_replay):
    




