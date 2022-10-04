'''
All experiments go here

Args
    none
Returns: pd.dataframe
'''
from evodm.learner import *

def evol_deepmind(num_evols = 1, N = 5, episodes = 50,
                  reset_every = 20, min_epsilon = 0.005, 
                  train_input = "fitness",  random_start = False, 
                  noise = False, noise_modifier = 1, num_drugs = 4, 
                  sigma = 0.5, normalize_drugs = True, 
                  player_wcutoff = 0.1, pop_wcutoff = 0.99, win_threshold = 200,
                  win_reward = 0, standard_practice = False, drugs = "none",
                  average_outcomes = False, mira = False, gamma = 0.99,
                  learning_rate = 0.0001, minibatch_size = 60, 
                  pre_trained = False, 
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

    #gotta modulate epsilon decay based on the number of episodes defined
    #0.005 = epsilon_decay^episodes
    hp.EPSILON_DECAY = pow(hp.MIN_EPSILON, 1/hp.EPISODES)

    if pre_trained and agent != "none":
        agent.master_memory = []
        rewards, agent, policy,V = practice(agent, naive=False, pre_trained = pre_trained)
        return [rewards, agent, policy]
        
    if mira:
        hp.NORMALIZE_DRUGS = False
        drugs = define_mira_landscapes()
    #initialize agent, including the updated hyperparameters
    agent = DrugSelector(hp = hp, drugs = drugs)
    naive_agent = deepcopy(agent) #otherwise it all gets overwritten by the actual agent
    if not average_outcomes:
        dp_agent = deepcopy(agent)
        dp_rewards, dp_agent, dp_policy, dp_V = practice(dp_agent, dp_solution = True, 
                                                   discount_rate= hp.DISCOUNT)

    #run the agent in the naive case and then in the reg case
    naive_rewards, naive_agent, naive_policy, V = practice(naive_agent, naive = True, standard_practice=standard_practice)
    rewards, agent, policy, V = practice(agent, naive = False)


    return [rewards, naive_rewards, agent, naive_agent, dp_agent, dp_rewards,
            dp_policy, naive_policy, policy, dp_V]

#rewards = evol_deepmind()
#naive_rewards= evol_deepmind(naive = True)

def define_mira_landscapes():
    '''
    Function to define the landscapes described in 
    Mira PM, Crona K, Greene D, Meza JC, Sturmfels B, Barlow M (2015) 
    Rational Design of Antibiotic Treatment Plans: A Treatment Strategy for Managing Evolution and Reversing Resistance. 
    PLoS ONE 10(5): e0122283. https://doi.org/10.1371/journal.pone.0122283
    '''
    drugs = []
    drugs.append([1.851, 2.082, 1.948, 2.434, 2.024, 2.198, 2.033, 0.034, 1.57, 2.165, 0.051, 0.083, 2.186, 2.322, 0.088, 2.821])    #AMP
    drugs.append([1.778, 1.782, 2.042, 1.752, 1.448, 1.544, 1.184, 0.063, 1.72, 2.008, 1.799, 2.005, 1.557, 2.247, 1.768, 2.047])    #AM
    drugs.append([2.258, 1.996, 2.151, 2.648, 2.396, 1.846, 2.23, 0.214, 0.234, 0.172, 2.242, 0.093, 2.15, 0.095, 2.64, 0.516])      #CEC
    drugs.append([0.16, 0.085, 1.936, 2.348, 1.653, 0.138, 2.295, 2.269, 0.185, 0.14, 1.969, 0.203, 0.225, 0.092, 0.119, 2.412])     #CTX
    drugs.append([0.993, 0.805, 2.069, 2.683, 1.698, 2.01, 2.138, 2.688, 1.106, 1.171, 1.894, 0.681, 1.116, 1.105, 1.103, 2.591])    #ZOX
    drugs.append([1.748, 1.7, 2.07, 1.938, 2.94, 2.173, 2.918, 3.272, 0.423, 1.578, 1.911, 2.754, 2.024, 1.678, 1.591, 2.923])       #CXM
    drugs.append([1.092, 0.287, 2.554, 3.042, 2.88, 0.656, 2.732, 0.436, 0.83, 0.54, 3.173, 1.153, 1.407, 0.751, 2.74, 3.227])       #CRO
    drugs.append([1.435, 1.573, 1.061, 1.457, 1.672, 1.625, 0.073, 0.068, 1.417, 1.351, 1.538, 1.59, 1.377, 1.914, 1.307, 1.728])    #AMC
    drugs.append([2.134, 2.656, 2.618, 2.688, 2.042, 2.756, 2.924, 0.251, 0.288, 0.576, 1.604, 1.378, 2.63, 2.677, 2.893, 2.563])    #CAZ
    drugs.append([2.125, 1.922, 2.804, 0.588, 3.291, 2.888, 3.082, 3.508, 3.238, 2.966, 2.883, 0.89, 0.546, 3.181, 3.193, 2.543])    #CTT
    drugs.append([1.879, 2.533, 0.133, 0.094, 2.456, 2.437, 0.083, 0.094, 2.198, 2.57, 2.308, 2.886, 2.504, 3.002, 2.528, 3.453])    #SAM
    drugs.append([1.743, 1.662, 1.763, 1.785, 2.018, 2.05, 2.042, 0.218, 1.553, 0.256, 0.165, 0.221, 0.223, 0.239, 1.811, 0.288])    #CPR
    drugs.append([0.595, 0.245, 2.604, 3.043, 1.761, 1.471, 2.91, 3.096, 0.432, 0.388, 2.651, 1.103, 0.638, 0.986, 0.963, 3.268])    #CPD
    drugs.append([2.679, 2.906, 2.427, 0.141, 3.038, 3.309, 2.528, 0.143, 2.709, 2.5, 0.172, 0.093, 2.453, 2.739, 0.609, 0.171])     #TZP
    drugs.append([2.59, 2.572, 2.393, 2.832, 2.44, 2.808, 2.652, 0.611, 2.067, 2.446, 2.957, 2.633, 2.735, 2.863, 2.796, 3.203])     #FEP
    return drugs

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
    
    

def signal2noise():
    '''
    somewhat out of place function to evolve a bunch of times at different noise_vec and store the true fitness against the noisy fitness
    Args
       none
    Returns: pd.dataframe
    '''

    noise_vec = [0,2,4,6,8,10,20,30,40,50,100]
    s_out = []
    n_out = []
    for i in iter(noise_vec):
        env=evol_env
    return 2

