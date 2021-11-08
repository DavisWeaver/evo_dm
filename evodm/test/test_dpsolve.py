import pytest
from evodm import dp_env, policy_improvement, value_iteration

@pytest.fixture
def env():
    return dp_env(N=4, sigma = 0.5)

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
