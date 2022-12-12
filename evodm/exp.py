from evodm.learner import *
from evodm.evol_game import define_mira_landscapes
from evodm.landscapes import Landscape
import pandas as pd
import numpy as np
from itertools import combinations

def evol_deepmind(num_evols = 1, N = 5, episodes = 50,
                  reset_every = 20, min_epsilon = 0.005, 
                  train_input = "fitness",  random_start = False, 
                  noise = False, noise_modifier = 1, num_drugs = 4, 
                  sigma = 0.5, normalize_drugs = True, 
                  player_wcutoff = 0.1, pop_wcutoff = 0.99, win_threshold = 200,
                  win_reward = 0, standard_practice = False, drugs = "none",
                  average_outcomes = False, mira = False, gamma = 0.99,
                  learning_rate = 0.0001, minibatch_size = 60, 
                  pre_trained = False, wf = False,
                  mutation_rate = 1e-5,
                  gen_per_step = 500,
                  pop_size = 10000,
                  agent = "none",
                  update_target_every = 310, total_resistance = False, 
                  starting_genotype = 0):
    """
    evol_deepmind is the main function that initializes and trains a learner to switch between n drugs
    to try and minimize the fitness of a population evolving on a landscape.

    ...

    Args
    ------
    num_evols: int
        how many evolutionary time steps are allowed to occur in between actions? defaults to one
    N: int
        number of alleles on the fitness landscape - landscape size scales 2^N. defaults to 5
    sigma: float
        numeric value determing the degree of epistasis in the underlying landscapes
    episodes: int
        number of episodes to train over
    reset_every: int
        number of evolutionary steps per episode
    min_epsilon: float
        epsilon at which we will stop training. the epsilon_decay hyperparameter is automatically 
        modulated based on this parameter and the number of episodes you want to run
    train_input: string
        should we use the state vector (genotype), or the fitness vector (growth) to train the model?
        allowed values are 'state_vector' and 'fitness'
    random_start: bool
        Should the state vector be initialized (and re-initialized) with the population scattered across 
        the fitness landscape instead of concentrated in one place?
    noise: bool
        should we incorporate noise into the "sensor" readings that are fed into the learner
    num_drugs: int
        How many drugs should the agent have access to? defaults to 4
    normalize_drugs: bool
        should we normalize the landscapes that represent our drugs so that the 
        min value for fitness is 0 and the max is 1?
    drugs: list of arrays
        pre-specify landscapes for each drug option? if blank it will automatically compute them 
    mira: bool
        Should we use the E. Coli drug landscapes defined by Mira et al?
    player_wcutoff: float
        What fitness value should we use to define the player victory conditions?
    pop_wcutoff: float
        what fitness value should we use to define the population victory conditions?
    win_threshold: int
        how many consecutive actions does fitness need to remain beyond the 
        victory cutoff before the game is called?
    win_reward: float
        how much reward should be awarded or penalized based on the outcome of a given ep
    standard_practice: bool
        should the comparison group mimic the current standard practice? i.e. 
        should the comparison group give a random drug until fitness approaches 
        some maximum and then randomly switch to another available drug?
    average_outcomes: bool
    gamma: float
        discount rate
    learning_rate: float
    """
    #initialize hyperparameters - and edit them according to the user inputs
    hp = hyperparameters()
    hp.NUM_EVOLS = int(num_evols)
    hp.N = int(N)
    hp.EPISODES = int(episodes)
    hp.MIN_EPSILON = min_epsilon
    hp.RESET_EVERY = int(reset_every)
    hp.TRAIN_INPUT = train_input
    hp.RANDOM_START = random_start
    hp.NOISE = noise
    hp.NOISE_MODIFIER = noise_modifier
    hp.NUM_DRUGS = int(num_drugs)
    hp.SIGMA = sigma
    hp.NORMALIZE_DRUGS = normalize_drugs
    hp.PLAYER_WCUTOFF = player_wcutoff
    hp.POP_WCUTOFF = pop_wcutoff
    hp.WIN_THRESHOLD = int(win_threshold)
    hp.WIN_REWARD = win_reward
    hp.AVERAGE_OUTCOMES = average_outcomes
    hp.DISCOUNT = gamma
    hp.LEARNING_RATE = learning_rate
    hp.MIRA = mira
    hp.MINIBATCH_SIZE = int(minibatch_size)
    hp.UPDATE_TARGET_EVERY = int(update_target_every)
    hp.TOTAL_RESISTANCE = total_resistance
    hp.STARTING_GENOTYPE = int(starting_genotype)
    hp.WF = wf
    hp.MUTATION_RATE = mutation_rate
    hp.GEN_PER_STEP = gen_per_step
    hp.POP_SIZE = pop_size

    #gotta modulate epsilon decay based on the number of episodes defined
    #0.005 = epsilon_decay^episodes
    hp.EPSILON_DECAY = pow(hp.MIN_EPSILON, 1/hp.EPISODES)

    if pre_trained and agent != "none":
        agent.master_memory = []
        rewards, agent, policy,V = practice(agent, naive=False, wf = wf, pre_trained = pre_trained)
        return [rewards, agent, policy]
        
    if mira:
        hp.NORMALIZE_DRUGS = False
        drugs = define_mira_landscapes()
    #initialize agent, including the updated hyperparameters
    agent = DrugSelector(hp = hp, drugs = drugs)
    naive_agent = deepcopy(agent) #otherwise it all gets overwritten by the actual agent
    if not any([average_outcomes, wf]):
        dp_agent = deepcopy(agent)
        dp_rewards, dp_agent, dp_policy, dp_V = practice(dp_agent, dp_solution = True, 
                                                         discount_rate= hp.DISCOUNT)

    #run the agent in the naive case and then in the reg case
    naive_rewards, naive_agent, naive_policy, V = practice(naive_agent, naive = True, 
                                                           standard_practice=standard_practice, 
                                                           wf=wf)
    rewards, agent, policy, V = practice(agent, naive = False, wf = wf)
    if wf:
        dp_policy=[]
        dp_agent=[]
        dp_V = []
        dp_rewards=[]

    return [rewards, naive_rewards, agent, naive_agent, dp_agent, dp_rewards,
            dp_policy, naive_policy, policy, dp_V]

#rewards = evol_deepmind()
#naive_rewards= evol_deepmind(naive = True)

def mdp_mira_sweep(num_evals, episodes = 10, num_steps = 20, normalize_drugs = False):
    '''
    Function to evaluate performance of the MDP optimal for different discount_rates (gamma)
    Args:
        num_evals: int
            how many gamma parameters to test
        episodes: int
            how many episodes should be evaluated per gamma parameter
    '''
    hp = hyperparameters()
    hp.EPISODES = episodes
    hp.RESET_EVERY = num_steps
    hp.N = 4
    hp.NUM_DRUGS = 15
    hp.NORMALIZE_DRUGS = normalize_drugs

    drugs = define_mira_landscapes()
    agent = DrugSelector(hp = hp, drugs = drugs)

    discount_range = np.linspace(0.0001, 0.999, num = num_evals)
    mem_list = []
    policy_list = []

    for i in iter(discount_range):
        rewards_i, agent_i, policy_i, V = practice(deepcopy(agent), dp_solution = True, discount_rate = i)
        mem_i = agent_i.master_memory
        mem_list.append([mem_i, i])
        policy_list.append([policy_i, i])

    return [mem_list, policy_list]

def mdp_sweep(N, sigma_range = [0,2], num_drugs_max=20, episodes=10, num_steps=20,
              normalize_drugs=True, num_evals=10, replicates = 3):
    '''
    Function to evaluate performance of the MDP optimal for different parameter regimes
        N: int
        sigma_range: list
            epistasis constant
        num_drugs_max: int
        episodes: int
            how many episodes should be evaluated per parameter regime
        num_steps: int
            episode length
        normalize_drugs: bool
        num_evals: int
            number of parameters to eval between min,max defined by sigma_range and num_drugs_range
        
    returns list
    '''

    sigma_range = np.linspace(sigma_range[0], sigma_range[1], num=num_evals)
    num_drugs_range = [i+1 for i in range(num_drugs_max)]

    mem_list_dp = []
    policy_list_dp = []
    mem_list_random = []

    for i in iter(sigma_range):
        for j in iter(num_drugs_range):
            #define new drug selector class for each of these scenarios
            hp = hyperparameters()
            hp.EPISODES = episodes
            hp.RESET_EVERY = num_steps
            hp.N = N
            hp.NUM_DRUGS = j
            hp.NORMALIZE_DRUGS = normalize_drugs
            hp.SIGMA = i
            agent_i = DrugSelector(hp = hp)
            for z in range(replicates):
                #Solve the MDP
                rewards_i, agent_dp, policy_dp, V = practice(deepcopy(agent_i), dp_solution = True)
                mem_i = agent_dp.master_memory
                mem_list_dp.append([mem_i, i, j, z])
                policy_list_dp.append([policy_dp, i, j, z])

                #Do it for a random
                rewards_i, agent_random, policy_random, V = practice(deepcopy(agent_i), naive=True)
                mem_i = agent_random.master_memory
                mem_list_random.append([mem_i, i, j, z])

    return [mem_list_dp, policy_list_dp, mem_list_random]
    
def test_generic_policy(policy, agent, 
                        prev_action = False):
    '''
    Function to test a generic policy for performance
    Args:
        policy: list of lists
            policy matrix specifying deterministic action selection for each state-evolstep. 
        episodes: int
        num_steps: int
        normalize_drugs= bool
    
    returns list of lists
        agent memory specifying performance.
    '''
    hp = agent.hp
    drugs = agent.env.drugs
    
    clean_agent = DrugSelector(hp = hp, drugs = drugs)

    rewards, out_agent, policy, V = practice(deepcopy(clean_agent), dp_solution=True, 
                                     policy = policy, prev_action = prev_action)

    mem = out_agent.master_memory
    return mem

def sweep_replicate_policy(agent):
    '''
    Function to sweep the policy learned by a given replicate at every episode
    Args:
        episodes: int
            how many episodes should be evaluated per policy
        normalize_drugs: bool
    '''    
    policies = agent.policies

    mem_list = []
    for i in range(len(policies)): 
        policy = policies[i][0]
        mem_i = test_generic_policy(policy, agent = agent,
                                    prev_action = True)
        mem_list.append(mem_i)
    
    return mem_list

#sweep through all two-drug policy combinations of the mira landscapes
def policy_sweep(episodes, normalize_drugs = False, num_steps = 20):
    '''
    Function to sweep through all two-drug policy combinations and test performance in simulated system
    Args:
        episodes: int
            how many episodes should be evaluated per policy
        normalize_drugs: bool
    '''
    hp = hyperparameters()
    hp.EPISODES = episodes
    hp.RESET_EVERY = num_steps
    hp.N = 4
    hp.NUM_DRUGS = 15
    hp.NORMALIZE_DRUGS = normalize_drugs

    drugs = define_mira_landscapes()
    
    #git iterable for all drugs
    all_drugs = [i for i in range(len(drugs))]

    #grab all possible combinations of two drugs
    all_comb = [i for i in combinations(all_drugs, 2)]

    mem_list = []
    for j in range(hp.N**2):
        hp.STARTING_GENOTYPE = j
        agent = DrugSelector(hp = hp, drugs = drugs)
        for i in iter(all_comb):
            policy_i = convert_two_drug(drug_comb = i, num_steps = num_steps, num_drugs = 15)
            rewards_i, agent_i, policy, V = practice(deepcopy(agent), dp_solution = True, policy=policy_i)
            mem_i = agent_i.master_memory
            mem_list.append([mem_i, i, j])
        
    return mem_list

def convert_two_drug(drug_comb, num_steps = 20, num_drugs = 15, N = 4):
    '''
    Function to convert two-drug combo to an alternating policy of the same form as compute_optimal
    Args:
        drug_comb: tuple
    '''
    #grab what the row should look like
    row = list((drug_comb*100))[:num_steps]
    policy = []
    #repeat for every state since these are going to be state independent
    for i in range(N**2):
        policy.append(row)

    return policy
    
    

def signal2noise(noise_vec):
    '''
    somewhat out of place function to evolve a bunch of times at different noise_vec and store the true fitness against the noisy fitness
    Args
       none
    Returns: pd.dataframe
    '''
    drugs = define_mira_landscapes()
    #set up outer loop
    df = pd.DataFrame({'fitness':[], 'noisy_fitness':[], 'noise_modifier':[]})
    for i in iter(noise_vec):
        s_out = [] #set up inner loop
        n_out = []
        env=evol_env(N=4, drugs = drugs, noise_modifier = i, win_threshold=10000, num_drugs=15,normalize_drugs=False)
        for j in range(10000):
            env.action = random.randint(np.min(env.ACTIONS),np.max(env.ACTIONS))
            env.step()
            s_out.append(float(env.fitness)) #Extra indexing to escape the data structures
            n_out.append(env.sensor_fitness)
        #payoff inner loop
        df_i = pd.DataFrame({'fitness':s_out, 'noisy_fitness':n_out})
        df_i = df_i.assign(noise_modifier = i)
        df = pd.concat([df, df_i])
    
    return df

def count_jumps(gen_per_step = 50, pop_size=10000):
    '''
    experiment to count the number of jumps that the E.Coli population can achieve given gen_per_step and pop_size 
    Args
       gen_per_step
       pop_size
    Returns: pd.dataframe
    '''
    hp_wf = hyperparameters()
    hp_wf.WF = True
    hp_wf.EPISODES=5
    hp_wf.MIN_REPLAY_MEMORY_SIZE=50
    hp_wf.MINIBATCH_SIZE = 25
    hp_wf.TRAIN_INPUT = 'fitness'
    hp_wf.GEN_PER_STEP = gen_per_step
    hp_wf.POP_SIZE = pop_size

    agent = DrugSelector(hp = hp_wf)
    jumps = []
    for i in range(1000):
        agent.env.update_drug(random.randint(np.min(agent.env.ACTIONS), np.max(agent.env.ACTIONS)))
        agent.env.step()
        genotypes = list(agent.env.pop.keys())
        num_jumps = [i.count('1') for i in genotypes]
        jumps.append(np.max(num_jumps))
        agent.env.reset()


def compute_opp_ls(drugids = ['CTX', 'CPR', 'SAM', 'AMP', 'TZP']):
    drugs= define_mira_landscapes(as_dict= True)
    opp_ls = [np.min([drugs[i][j] for i in iter(drugids)]) for j in range(16)]
    
    return opp_ls



