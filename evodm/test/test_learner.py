from evodm.learner import compute_optimal_policy, compute_optimal_action, DrugSelector, practice, hyperparameters
from evodm.dpsolve import backwards_induction, dp_env
from evodm.exp import define_mira_landscapes
import random
import numpy as np
import pytest
from itertools import chain
from copy import deepcopy
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
    hp.AVERAGE_OUTCOMES = True
    return hp

@pytest.fixture
def hp_mira():
    hp = hyperparameters()
    hp.TRAIN_INPUT = "fitness"
    hp.MIN_REPLAY_MEMORY_SIZE = 20
    hp.MINIBATCH_SIZE = 10
    hp.RESET_EVERY = 20
    hp.EPISODES = 10
    hp.N = 4
    hp.NUM_DRUGS = 15
    hp.NOISE=True
    hp.NOISE_MODIFIER=5
    hp.AVERAGE_OUTCOMES = False
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
    hp_state.AVERAGE_OUTCOMES = True
    return hp_state

@pytest.fixture
def hp_one_traj():
    hp_one_traj = hyperparameters()
    hp_one_traj.TRAIN_INPUT = "state_vector"
    hp_one_traj.MIN_REPLAY_MEMORY_SIZE = 20
    hp_one_traj.MINIBATCH_SIZE = 10
    hp_one_traj.RESET_EVERY = 20
    hp_one_traj.EPISODES = 10
    hp_one_traj.AVERAGE_OUTCOMES = False
    hp_one_traj.N = 4
    return hp_one_traj

@pytest.fixture
def hp_one_traj_fitness():
    hp_one_traj_fitness = hyperparameters()
    hp_one_traj_fitness.TRAIN_INPUT = "fitness"
    hp_one_traj_fitness.MIN_REPLAY_MEMORY_SIZE = 20
    hp_one_traj_fitness.MINIBATCH_SIZE = 20
    hp_one_traj_fitness.RESET_EVERY = 10
    hp_one_traj_fitness.EPISODES = 10
    hp_one_traj_fitness.AVERAGE_OUTCOMES = False
    hp_one_traj_fitness.N = 4
    return hp_one_traj_fitness

@pytest.fixture
def hp_N5():
    hp_N5 = hyperparameters()
    hp_N5.TRAIN_INPUT = "fitness"
    hp_N5.MIN_REPLAY_MEMORY_SIZE = 20
    hp_N5.MINIBATCH_SIZE = 20
    hp_N5.RESET_EVERY = 10
    hp_N5.EPISODES = 10
    hp_N5.AVERAGE_OUTCOMES = False
    hp_N5.N = 5
    return hp_N5

@pytest.fixture
def hp_default():
    hp_default = hyperparameters()
    hp_default.NUM_EVOLS = 1
    hp_default.N = 5
    hp_default.EPISODES = 50
    hp_default.RESET_EVERY = 200
    hp_default.MIN_EPSILON = 0.005
    hp_default.TRAIN_INPUT = "fitness"
    hp_default.RANDOM_START = False
    hp_default.NOISE = False
    hp_default.NOISE_MODIFIER = 1
    hp_default.NUM_DRUGS = 4
    hp_default.SIGMA = 0.5
    hp_default.NORMALIZE_DRUGS = True
    hp_default.PLAYER_WCUTOFF = 0.1
    hp_default.POP_WCUTOFF = 0.99
    hp_default.WIN_THRESHOLD = 200
    hp_default.WIN_REWARD = 0
    hp_default.AVERAGE_OUTCOMES = False
    return hp_default

@pytest.fixture
def hp_5evols():
    hp_5evols = hyperparameters()
    hp_5evols.N = 5
    hp_5evols.NUM_DRUGS = 5
    hp_5evols.RESET_EVERY = 20
    hp_5evols.NUM_EVOLS = 5
    hp_5evols.TRAIN_INPUT = "fitness"
    hp_5evols.EPISODES = 4
    hp_5evols.AVERAGE_OUTCOMES = False
    return hp_5evols

@pytest.fixture
def hp_small():
    hp_small = hyperparameters()
    hp_small.DENSE = True
    hp_small.N=3
    hp_small.NUM_EVOLS = 1
    hp_small.NUM_DRUGS=5
    hp_small.TRAIN_INPUT = "fitness"
    hp_small.MIN_REPLAY_MEMORY_SIZE = 20
    hp_small.MINIBATCH_SIZE = 10
    hp_small.RESET_EVERY = 20
    hp_small.EPISODES = 1000
    return hp_small

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
def ds_one_traj(hp_one_traj):
    ds_one_traj = DrugSelector(hp=hp_one_traj)
    for episode in range(1, ds_one_traj.hp.EPISODES + 1):
        for i in range(1, ds_one_traj.hp.RESET_EVERY):
            ds_one_traj.env.action = random.randint(np.min(ds_one_traj.env.ACTIONS),np.max(ds_one_traj.env.ACTIONS))
            ds_one_traj.env.step()
            ds_one_traj.update_replay_memory()
        ds_one_traj.env.reset()
    return ds_one_traj

@pytest.fixture
def ds_one_traj_fitness(hp_one_traj_fitness):
    ds_one_traj_fitness = DrugSelector(hp=hp_one_traj_fitness)
    for episode in range(1, ds_one_traj_fitness.hp.EPISODES + 1):
        for i in range(1, ds_one_traj_fitness.hp.RESET_EVERY):
            ds_one_traj_fitness.env.action = random.randint(np.min(ds_one_traj_fitness.env.ACTIONS),np.max(ds_one_traj_fitness.env.ACTIONS))
            ds_one_traj_fitness.env.step()
            ds_one_traj_fitness.update_replay_memory()
        ds_one_traj_fitness.env.reset()
    return ds_one_traj_fitness

@pytest.fixture
def ds_N5(hp_N5):
    ds_N5 = DrugSelector(hp=hp_N5)
    for episode in range(1, ds_N5.hp.EPISODES + 1):
        for i in range(1, ds_N5.hp.RESET_EVERY):
            ds_N5.env.action = random.randint(np.min(ds_N5.env.ACTIONS),np.max(ds_N5.env.ACTIONS))
            ds_N5.env.step()
            ds_N5.update_replay_memory()
        ds_N5.env.reset()
    return ds_N5

@pytest.fixture
def ds_evol5(hp_5evols):
    ds_evol5 = DrugSelector(hp=hp_5evols)
    for episode in range(1, ds_evol5.hp.EPISODES + 1):
        for i in range(1, ds_evol5.hp.RESET_EVERY):
            ds_evol5.env.action = random.randint(np.min(ds_evol5.env.ACTIONS),np.max(ds_evol5.env.ACTIONS))
            ds_evol5.env.step()
            ds_evol5.update_replay_memory()
        ds_evol5.env.reset()
    return ds_evol5

@pytest.fixture
def ds_default(hp_N5):
    ds_default = DrugSelector(hp=hp_N5)
    for episode in range(1, ds_default.hp.EPISODES + 1):
        for i in range(1, ds_default.hp.RESET_EVERY):
            ds_default.env.action = random.randint(np.min(ds_default.env.ACTIONS),np.max(ds_default.env.ACTIONS))
            ds_default.env.step()
            ds_default.update_replay_memory()
        ds_default.env.reset()
    return ds_default


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
def ds_mira_noise(hp_mira):
    drugs = define_mira_landscapes()
    hp.NUM_DRUGS = 15
    hp.RESET_EVERY = 20
    hp.NORMALIZE_DRUGS = False
    ds = DrugSelector(hp = hp_mira, drugs = drugs)
    for episode in range(1, ds.hp.EPISODES + 1):
        for i in range(1, ds.hp.RESET_EVERY):
            ds.env.action = random.randint(np.min(ds.env.ACTIONS),np.max(ds.env.ACTIONS))
            ds.env.step()
            ds.update_replay_memory()
        ds.env.reset()
    return ds

@pytest.fixture
def ds_small(hp_small):
    ds_replay = DrugSelector(hp= hp_small)
    x,ds_small,y, V = practice(ds_replay, naive = True)
    return ds_small

@pytest.fixture 
def ds_mira():
    hp = hyperparameters()
    hp.N = 4
    drugs = define_mira_landscapes()
    hp.NUM_DRUGS = 15
    hp.RESET_EVERY = 20
    hp.NORMALIZE_DRUGS = False
    ds_mira = DrugSelector(hp = hp, drugs = drugs)
    return ds_mira

@pytest.fixture
def allow_mat():
    allow_mat = {
        0:(0,1,2,4),
        1:(0,1,3,5),
        2:(0,2,3,6),
        3:(1,2,3,7),
        4:(0,4,5,6),
        5:(1,4,5,7),
        6:(2,4,6,7),
        7:(3,5,6,7)
    }
    #000 -> 100 (4), 010 (2), 001 (1), 000 (4)
    #001 -> 000 (0), 011 (3), 001 (1), 101 (5)
    #010 -> 000(0), 010 (2), 110 (6), 011 (3)
    #011 -> 010 (2), 111(7), 001 (1), 011 (3)
    #100 -> 100 (4), 000 (0), 110 (6), 101(5)
    #101 -> 101 (5), 111 (7), 100 (4), 001 (1)
    #110 -> 110 (6), 111 (7), 100 (4), 010 (2)
    #111 -> 111 (7), 110 (6), 101 (5), 011 (3)
    return allow_mat


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

@pytest.fixture
def current_states_onetraj(ds_one_traj):
    minibatch = random.sample(ds_one_traj.replay_memory, ds_one_traj.hp.MINIBATCH_SIZE)
    current_states, new_current_states = ds_one_traj.get_current_states(minibatch = minibatch)
    return [current_states, new_current_states]

def test_replay_noise(ds_mira_noise):
    last = ds_mira_noise.replay_memory[len(ds_mira_noise.replay_memory)-1][3][15]
    assert ds_mira_noise.env.sensor_fitness == last

def test_replay_noise2(ds_mira_noise):
    last = ds_mira_noise.replay_memory[len(ds_mira_noise.replay_memory)-1][3][15]
    assert ds_mira_noise.env.fitness != last

def test_replay_noise4(ds_mira_noise):
    last = ds_mira_noise.master_memory[len(ds_mira_noise.master_memory)-1][2][3][15]
    assert ds_mira_noise.env.sensor_fitness == last

def test_replay_noise5(ds_mira_noise):
    bools= [i[4] >= 0 for i in ds_mira_noise.master_memory]
    assert np.all(bools)

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

#make sure the entire population is in one state for current_states_onetraj
def test_current_states_onetraj(current_states_onetraj):
    current_states = current_states_onetraj[0]
    bools = [np.max(i) == 1 for i in current_states]
    assert all(bools)

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

def test_predict_qs_onetraj(current_states_onetraj, ds_one_traj):
    current_states = current_states_onetraj[0]
    current_qs_list = ds_one_traj.model.predict(current_states)
    
    #this line flattens the list
    qs = list(chain.from_iterable(current_qs_list))
    bools = [isinstance(i, np.floating) for i in qs] #np type floats don't work with regular python type checking - super strange

    assert all(bools)

def test_get_qs(ds_replay):
    for i in range(3):
        ds_replay.env.step()
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
        ds_replay.env.step()
        ds_replay.update_replay_memory()
    
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


def test_get_qs3(ds_N5):
    for i in range(3):
        ds_N5.env.step()
    qs = ds_N5.get_qs()
    assert len(qs) == len(ds_N5.env.ACTIONS)
    
#now with full default hyperparameters
def test_get_qs4(ds_default):
    for i in range(3):
        ds_default.env.step()
    qs = ds_default.get_qs()
    assert len(qs) == len(ds_default.env.ACTIONS)

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

#test to make sure only legal moves are recorded in replay memory
def test_replay_memory(ds_small, allow_mat):
    states = [np.argmax(i[3]) for i in ds_small.master_memory]
    states_ep = [states[i:i+19] for i in range(0, len(states), 19)]
    bool_list = []
    for i in range(len(states_ep)):
        bool_list_j = []
        for j in range(len(states_ep[i])):
            if j != len(states_ep[i])-1: #don't do this step for the last step in the counter
                bool_j = states_ep[i][j+1] in allow_mat[states_ep[i][j]]
                bool_list_j.append(bool_j)
            bool_list.append(np.all(bool_list_j))
    

    assert np.all(bool_list)

@pytest.fixture
def opt_policy(ds_one_traj):
    opt_policy,x = compute_optimal_policy(ds_one_traj)
    return opt_policy

def test_compute_optimal_policy(opt_policy):
    bools = []
    for i in range(len(opt_policy)):
        bools_i = [j in [0,1,2,3] for j in opt_policy[i]]
        bools.append(bools_i)
    assert all(bools)

def test_compute_optimal_action(ds_one_traj, opt_policy):
    action = compute_optimal_action(agent = ds_one_traj, policy = opt_policy, step = 2)
    assert action in [1,2,3,4]

#test that we are getting a policy back when we use this for a state_vector trained RL
def test_compute_implied_policy(ds_one_traj):
    policy = ds_one_traj.compute_implied_policy(update = False)
    bools = [np.sum(i) == 1 for i in policy]
    assert all(bools)

def test_practice(ds_one_traj_fitness):
    reward, agent, policy, V = practice(ds_one_traj_fitness, dp_solution=True)
    reward, agent, policy, V = practice(ds_one_traj_fitness, naive=True)


def test_mira_practice(ds_mira):
    reward, agent, policy, V = practice(ds_mira, dp_solution=True)
    reward, agent, policy, V = practice(ds_mira, naive=True)

#narrow down on the issue
def test_compute_optimal_policy(ds_mira):
    policy,x = compute_optimal_policy(ds_mira, discount_rate= 0.001)
    policy2,x = compute_optimal_policy(ds_mira, discount_rate = 0.999)
    bools_list = []
    for s in range(len(policy2)):
        #this checks for equivalence of policy for 
        bools = [policy[s][j] != policy2[s][j] for j in range(len(policy2[s]))]
        bools_list.append(bools)
    bools_list = list(chain.from_iterable(bools_list))
    
    assert any(bools_list)



@pytest.fixture
def mira_env():
    drugs = define_mira_landscapes()
    return dp_env(N=4, num_drugs = 15, drugs = drugs, sigma = 0.5)

@pytest.fixture
def hp_wf():
    hp_wf = hyperparameters()
    hp_wf.WF = True
    hp_wf.EPISODES=5
    hp_wf.MIN_REPLAY_MEMORY_SIZE=50
    hp_wf.MINIBATCH_SIZE = 25
    return hp_wf

def test_init_wf(hp_wf):
    #don't need to assert anything - just make sure it doesn't break when we do this
    agent = DrugSelector(hp = hp_wf)
    
def test_train_wf(hp_wf):
    #again asserting things is for losers - no errors = cool
    agent = DrugSelector(hp = hp_wf)
    for i in range(100):
        agent.env.action = random.randint(np.min(agent.env.ACTIONS), np.max(agent.env.ACTIONS))
        agent.env.step()
        agent.update_replay_memory()
    
    agent.train()

def test_practice_wf(hp_wf):
    agent = DrugSelector(hp=hp_wf)
    reward_list, agent, policy, v = practice(
        agent = agent, wf = True
    )

def test_practice_wf_fit(hp_wf):
    hp_wf.train_input = 'fitness'
    agent = DrugSelector(hp=hp_wf)
    reward_list, agent, policy, v = practice(
        agent = agent, wf = True
    )

def test_compute_implied_policy_wf(hp_wf):
    agent = DrugSelector(hp = hp_wf)
    for i in range(1000):
        agent.env.action = random.randint(np.min(agent.env.ACTIONS), np.max(agent.env.ACTIONS))
        agent.env.step()
        agent.update_replay_memory()
    
    agent.train()
    agent.compute_implied_policy(update = True)

def test_memory_wf(hp_wf):
    hp_wf.TRAIN_INPUT = 'fitness'
    agent = DrugSelector(hp = hp_wf)
    for i in range(10):
        agent.env.action = random.randint(np.min(agent.env.ACTIONS), np.max(agent.env.ACTIONS))
        agent.env.step()
        agent.update_replay_memory()

    assert isinstance(agent.master_memory[0][4], float)

def test_memory_wf(hp_wf):
    hp_wf.TRAIN_INPUT = 'state_vector'
    agent = DrugSelector(hp = hp_wf)
    for i in range(10):
        agent.env.action = random.randint(np.min(agent.env.ACTIONS), np.max(agent.env.ACTIONS))
        agent.env.step()
        agent.update_replay_memory()

    assert isinstance(agent.master_memory[0][3], float)

def test_memory_wf2(hp_wf):
    hp_wf.TRAIN_INPUT = 'fitness'
    agent = DrugSelector(hp = hp_wf)
    for i in range(10):
        agent.env.update_drug(random.randint(np.min(agent.env.ACTIONS), np.max(agent.env.ACTIONS)))
        agent.env.step()
        agent.update_replay_memory()

    fit = agent.env.compute_pop_fitness(drug = agent.env.drug, sv = agent.env.pop)

    assert fit == agent.env.fitness

def test_memory_wf3(hp_wf):
    hp_wf.TRAIN_INPUT = 'fitness'
    agent = DrugSelector(hp = hp_wf)
    for i in range(10):
        agent.env.update_drug(random.randint(np.min(agent.env.ACTIONS), np.max(agent.env.ACTIONS)))
        agent.env.step()
        agent.update_replay_memory()

    fit = agent.env.compute_pop_fitness(drug = agent.env.drug, sv = agent.env.pop)
    assert fit == agent.replay_memory[len(agent.replay_memory) - 1][3][15]

def test_memory_wf4(hp_wf):
    hp_wf.TRAIN_INPUT = 'fitness'
    hp_wf.GEN_PER_STEP = 50
    agent = DrugSelector(hp = hp_wf)
    for i in range(10):
        agent.env.update_drug(random.randint(np.min(agent.env.ACTIONS), np.max(agent.env.ACTIONS)))
        agent.env.step()
        agent.update_replay_memory()
