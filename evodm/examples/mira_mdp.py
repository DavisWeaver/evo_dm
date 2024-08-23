from evodm.dpsolve import dp_env, backwards_induction, value_iteration, policy_iteration
from evodm import define_mira_landscapes, evol_env
import numpy as np


def mira_env(compute_P = False):
    drugs = define_mira_landscapes()
    envdp = dp_env(N=4, num_drugs = 15, drugs = drugs, sigma = 0.5,
                  compute_P=compute_P)
    env = evol_env(N=4, drugs = drugs, num_drugs = 15, normalize_drugs=False,
                   train_input = 'fitness')
    return envdp, env

envdp, env = mira_env()

#generate drug sequences using policies from backwards induction,
#value iteration, or policy iteration
def get_sequences(policy, num_episodes=100, episode_length = 20):
    ep_number =[]
    opt_drug = []
    time_step = []
    for i in range(num_episodes):
        env.reset()
        for j in range(episode_length):
            action_opt = policy[np.argmax(env.state_vector)]
            env.action = action_opt
            env.step()
            
            #save the optimal drug, time step, and episode number
            opt_drug.append(env.action)
            time_step.append(j)
            ep_number.append(i) 
   

# Solve the MDP using different algorithms
policy_bi, V_bi = backwards_induction(envdp)

env = mira_env(compute_P = True)

policy_vi, V_vi = value_iteration(envdp)

policy_pi, V_pi = policy_iteration(envdp)



