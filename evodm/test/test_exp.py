from evodm.exp import mdp_mira_sweep
from itertools import chain

def test_mdp_mira_sweep():
    mem_list = mdp_mira_sweep(num_evals = 10)[0]
    assert len(mem_list) == 10

#test the policies are actually different based on gamma
def test_mdp_mira_sweep():
    policies = mdp_mira_sweep(num_evals = 2, num_steps= 20, episodes = 1)[1]
    policy = policies[0][0]
    policy2 = policies[1][0]
    bools_list = []
    for s in range(len(policy2)):
        #this checks for equivalence of policy for 
        bools = [policy[s][j] != policy2[s][j] for j in range(len(policy2[s]))]
        bools_list.append(bools)
    bools_list = list(chain.from_iterable(bools_list))
    assert any(bools_list)