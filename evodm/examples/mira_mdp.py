from evodm.dpsolve import dp_env, backwards_induction, value_iteration, policy_iteration
from evodm import define_mira_landscapes

def mira_env(compute_P = False):
    drugs = define_mira_landscapes()
    return dp_env(N=4, num_drugs = 15, drugs = drugs, sigma = 0.5,
                  compute_P=compute_P)

env = mira_env()

# Solve the MDP using different algorithms
policy_bi, V_bi = backwards_induction(env)

env = mira_env(compute_P = True)

policy_vi, V_vi = value_iteration(env)

policy_pi, V_pi = policy_iteration(env)