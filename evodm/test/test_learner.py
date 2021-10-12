from evodm import DrugSelector, practice, hyperparameters
import random
import numpy as np
import pytest

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
def ds(hp):
    ds = DrugSelector(hp = hp)
    return ds

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

def test_get_qs(ds_replay):
    qs = ds_replay.get_qs()
    assert len(qs) == len(ds_replay.env.ACTIONS)

def test_get_qs2(ds_replay,ds):
    for i in range(3):
        ds.env.step()
        ds.update_replay_memory()
    
    #just make sure q changes based on the inputs
    qs1 = ds.get_qs()
    qs2 = ds_replay.get_qs()

    bools = [qs1[i] != qs2[i] for i in range(len(qs1))]
    assert any(bools)
    
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
    




