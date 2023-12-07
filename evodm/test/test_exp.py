from evodm.exp import mdp_mira_sweep, evol_deepmind
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

def test_evol_deepmind_phenom():
    #make sure the phenom landscapes don't crash everything
    out = evol_deepmind(train_input = 'fitness', episodes=5, num_drugs=15, N=4, 
                        mira=True, normalize_drugs=False, phenom=20)

def test_evol_deepmind_wf():
    #just make sure it runs quietly at a minimum
    out = evol_deepmind(wf = True, train_input='fitness', episodes=5, 
                        num_drugs = 15, N=4, mira=True, 
                        normalize_drugs=False)
                

def test_evol_deepmind_wf2():
    #just make sure it runs quietly at a minimum
    out = evol_deepmind(wf = True, train_input='fitness', episodes=5, 
                        num_drugs = 15, N=4, mira=True, 
                        normalize_drugs=False)

def test_evol_deepmind_sparse():
    out = evol_deepmind(dense= False, train_input='fitness', episodes=5, reset_every=10)
    agent = out[2]
    assert agent.env.landscapes[0].get_TM().size < 4**agent.env.N
    assert len(agent.master_memory) == 50

def test_evol_deepmind_delay():
    out = evol_deepmind(train_input = "state_vector", delay = 1, mira=True, 
                        num_drugs=15, N=4, normalize_drugs=False, episodes = 5)