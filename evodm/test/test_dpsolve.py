import pytest
from evodm.dpsolve import dp_env, backwards_induction
from evodm import define_mira_landscapes
from itertools import chain

@pytest.fixture
def env():
    return dp_env(N=4, sigma = 0.5)

@pytest.fixture
def mira_env():
    drugs = define_mira_landscapes()
    return dp_env(N=4, num_drugs = 15, drugs = drugs, sigma = 0.5)

#make sure probs for every P[s][a] sum to 1
def test_P_probs(env):
    
    probs = []
    for s in range(env.nS):
        for a in range(env.nA):
            prob = 0
            for s_prime in range(len(env.P[s][a])):
                #0 because the transition tuples have the prob at first position
                prob += env.P[s][a][s_prime][0]
            probs.append(prob)
    bools = [i == 1.0 for i in probs]

    assert all(bools)

def test_backwards_induction(env):
    policy, V = backwards_induction(env)
    assert policy.shape == (16,20)

#test that changing gamma actually does something
def test_discount_rate(env):
    policy, V = backwards_induction(env, discount_rate=0.01)
    policy2,V2 = backwards_induction(env, discount_rate = 0.999)
    bools_list = []
    for s in range(len(policy2)):
        #this checks for equivalence of policy for 
        bools = [policy[s][j] != policy2[s][j] for j in range(len(policy2[s]))]
        bools_list.append(bools)
    bools_list = list(chain.from_iterable(bools_list))
    assert any(bools_list)

#Now for the mira condition which seems to be giving me trouble
def test_discount_mira(mira_env):
    policy, V = backwards_induction(mira_env, discount_rate=0.01)
    policy2,V2 = backwards_induction(mira_env, discount_rate = 0.999)
    bools_list = []
    for s in range(len(policy2)):
        #this checks for equivalence of policy for 
        bools = [policy[s][j] != policy2[s][j] for j in range(len(policy2[s]))]
        bools_list.append(bools)
    bools_list = list(chain.from_iterable(bools_list))
    assert any(bools_list)