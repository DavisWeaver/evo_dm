# this file will define the learner class, along with required methods -
# we are taking inspiration (and in some cases borrowing heavily) from the following
# tutorial: https://pythonprogramming.net/training-deep-q-learning-dqn-reinforcement-learning-python-tutorial/?completed=/deep-q-learning-dqn-reinforcement-learning-python-tutorial/
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv1D, MaxPooling1D, Flatten
from numpy.lib.utils import deprecate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from collections import deque
from evodm.evol_game import evol_env
from evodm.dpsolve import backwards_induction, dp_env
import random
import numpy as np 
import pandas as pd
from tqdm import tqdm
from copy import deepcopy
from itertools import combinations

# Function to set hyperparameters for the learner - just edit this any time you
# want to screw around with them.
#or edit directly


class hyperparameters:
    '''
    class to store the hyperparemeters that control evoDM
    ...
    Args
    ------
    self: class hyperparameters
    
    Returns class hyperparameters
    '''

    def __init__(self):
        # Model training settings
        self.REPLAY_MEMORY_SIZE = 10000
        self.MIN_REPLAY_MEMORY_SIZE = 1000
        self.MINIBATCH_SIZE = 100  
        self.UPDATE_TARGET_EVERY = 310 #every 500 steps, update the target
        self.TRAIN_INPUT = "state_vector"
        
        # Exploration settings
        self.DISCOUNT = 0.99  
        self.epsilon = 1  # lowercase because its not a constant
        self.EPSILON_DECAY = 0.95
        self.MIN_EPSILON = 0.001
        self.LEARNING_RATE = 0.002

        # settings control the evolutionary simulation
        self.NUM_EVOLS = 1 # how many evolutionary steps per time step
        self.SIGMA = 0.5
        self.NORMALIZE_DRUGS = True # should fitness values for all landscapes be bound between 0 and 1?
        self.AVERAGE_OUTCOMES = False #should we use the average of infinite evolutionary sims or use a single trajectory?
        # new evolutionary "game" every n steps or n *num_evols total evolutionary movements
        self.RESET_EVERY = 200
        self.EPISODES = 50
        self.N = 5
        self.RANDOM_START = False
        self.STARTING_GENOTYPE = 0 #default to starting at the wild type genotype
        self.NOISE = False #should the sensor readings be noisy?
        self.NOISE_MODIFIER = 1  #enable us to increase or decrease the amount of noise in the system
        self.NUM_DRUGS = 4
        self.MIRA = True

        #define victory conditions for player and pop
        self.PLAYER_WCUTOFF = 0.001
        self.POP_WCUTOFF = 0.999

        #define victory threshold
        self.WIN_THRESHOLD = 1000 # number of player actions before the game is called
        self.WIN_REWARD = 0

        # stats settings - 
        self.AGGREGATE_STATS_EVERY = 1  #agg every episode


# This is the class for the learning agent
class DrugSelector:
    
    def __init__(self, hp, drugs = "none"):
        '''
        Initialize the DrugSelector class
        ...
        Args
        ------
        self: class DrugSelector
        hp: class hyperparameters
            hyperparameters that control the evodm architecture and the 
            evolutionary simulations used to train it
        drugs: list of numeric matrices
            optional parameter - can pass in a list of drugs to use as the available actions. 
            If not provided, drugs will be procedurally generated


        Returns class DrugSelector
        '''
        # hp stands for hyperparameters
        self.hp = hp
        # initialize the environment
        self.env = evol_env(num_evols=self.hp.NUM_EVOLS, N = self.hp.N,
                            train_input= self.hp.TRAIN_INPUT, 
                            random_start=self.hp.RANDOM_START, 
                            num_drugs = self.hp.NUM_DRUGS, 
                            sigma=self.hp.SIGMA,
                            normalize_drugs = self.hp.NORMALIZE_DRUGS, 
                            win_threshold= self.hp.WIN_THRESHOLD, 
                            player_wcutoff = self.hp.PLAYER_WCUTOFF, 
                            pop_wcutoff= self.hp.POP_WCUTOFF,
                            win_reward=self.hp.WIN_REWARD, 
                            drugs = drugs, 
                            add_noise = self.hp.NOISE, 
                            noise_modifier= self.hp.NOISE_MODIFIER,
                            average_outcomes=self.hp.AVERAGE_OUTCOMES, 
                            starting_genotype = self.hp.STARTING_GENOTYPE)

        # main model  # gets trained every step
        self.model = self.create_model()

        # Target model this is what we .predict against every step
        self.target_model = self.create_model()
        self.target_model.set_weights(self.model.get_weights())

        self.replay_memory = deque(maxlen=self.hp.REPLAY_MEMORY_SIZE)
        self.master_memory = []
        self.target_update_counter = 0
        self.policies = []

    def create_model(self):

        model = Sequential()
        #need to change padding settings if using fitness to train model
        #because sequence may not be long enough
        if self.hp.TRAIN_INPUT == "state_vector":
            model.add(Conv1D(64, 3, activation="relu",
                         input_shape=self.env.ENVIRONMENT_SHAPE))
            model.add(Conv1D(64, 3, activation="relu"))
            model.add(MaxPooling1D(pool_size=2))
            model.add(Flatten())
        elif self.hp.TRAIN_INPUT == "fitness":
            #have to change the kernel size because of the weird difference in environment shape
            model.add(Dense(64, activation="relu",
                         input_shape=self.env.ENVIRONMENT_SHAPE))
        elif self.hp.TRAIN_INPUT == "pop_size":
            model.add(Conv1D(64, 3, activation="relu",
                         input_shape=self.env.ENVIRONMENT_SHAPE))
            model.add(Conv1D(64, 3, activation="relu"))
            model.add(MaxPooling1D(pool_size=2))
            model.add(Flatten())
        else:
            print("please specify either state_vector, fitness, or pop_size for train_input when initializing the environment")
            return
        model.add(Dropout(0.2))
        model.add(Dense(28, activation = "relu"))
        model.add(Dense(len(self.env.ACTIONS), activation="linear"))
        model.compile(loss="mse", optimizer=Adam(learning_rate=self.hp.LEARNING_RATE), metrics=['accuracy'])
        return model

    def update_replay_memory(self):

        if self.env.action_number !=1:
            self.replay_memory.append(self.env.sensor)
            #update master memory - for diagnostic purposes only
            if self.env.TRAIN_INPUT == "fitness":
                #want to save the state vector history somewhere, regardless of what we use for training. 
                self.master_memory.append([self.env.episode_number, self.env.action_number, self.env.sensor, self.env.state_vector])
            else:
                self.master_memory.append([self.env.episode_number, self.env.action_number, self.env.sensor])
      # Trains main network every step during episode
      #gonna chunk this out so I can actually test it
    def train(self):

        # Start training only if certain number of samples is already saved
        if len(self.replay_memory) < self.hp.MIN_REPLAY_MEMORY_SIZE:
            return

        # Get a minibatch of random samples from memory replay table
        minibatch = random.sample(self.replay_memory, self.hp.MINIBATCH_SIZE)

        #get the current states
        current_states, new_current_states = self.get_current_states(minibatch = minibatch)

        current_qs_list = self.model.predict(current_states)
        future_qs_list = self.target_model.predict(new_current_states)

        
        # Now we need to enumerate our batches
        X,y = self.enumerate_batch(minibatch = minibatch, future_qs_list = future_qs_list, 
                                   current_qs_list= current_qs_list)
                                   
        self.model.fit(X, y, batch_size=self.hp.MINIBATCH_SIZE, 
                       verbose=0, shuffle=False, callbacks=None)

        # If counter reaches set value, update target network with weights of main network
        if self.env.update_target_counter > self.hp.UPDATE_TARGET_EVERY:
            self.target_model.set_weights(self.model.get_weights())
            self.env.update_target_counter = 0

    #function to enumerate batch and generate X/y for training
    def enumerate_batch(self, minibatch, future_qs_list, current_qs_list):
        X = []
        y = []

        for index, (current_state, action, reward, new_current_state) in enumerate(minibatch):

            # If not a terminal state, get new q from future states, otherwise set it to 0
            # almost like with Q Learning, but we use just part of equation here
            max_future_q = np.max(future_qs_list[index])
            new_q = reward + self.hp.DISCOUNT * max_future_q

            # Update Q value for given state
            current_qs = current_qs_list[index]
            current_qs[action - 1] = new_q #again we need the minus 1 because of the dumb python indexing system

            # And append to our training data
            X.append(current_state)
            y.append(current_qs)

        # Fit on all samples as one batch, log only on terminal state
        #need to reshape x to match dimensions
        if self.env.TRAIN_INPUT == "state_vector":
            X = np.array(X).reshape(self.hp.MINIBATCH_SIZE, self.env.ENVIRONMENT_SHAPE[0], 
                                    self.env.ENVIRONMENT_SHAPE[1])
        else: 
            X = np.array(X).reshape(self.hp.MINIBATCH_SIZE, self.env.ENVIRONMENT_SHAPE[0])
        y = np.array(y)

        return X,y
        
    def get_current_states(self, minibatch):
        # Get current states from minibatch, then query NN model for Q values
        current_states = np.array([transition[0] for transition in minibatch])
        # Get future states from minibatch, then query NN model for Q values
        # When using target network, query it, otherwise main network should be queried
        new_current_states = np.array(
            [transition[3] for transition in minibatch])

        #reshape to match expected input dimensions
        if self.env.TRAIN_INPUT == "state_vector":
            current_states = current_states.reshape(self.hp.MINIBATCH_SIZE, 
                                                    self.env.ENVIRONMENT_SHAPE[0],
                                                    self.env.ENVIRONMENT_SHAPE[1])

            new_current_states = new_current_states.reshape(self.hp.MINIBATCH_SIZE, 
                                                            self.env.ENVIRONMENT_SHAPE[0],
                                                            self.env.ENVIRONMENT_SHAPE[1])               
        else:
            current_states.reshape(self.hp.MINIBATCH_SIZE, 
                                   self.env.ENVIRONMENT_SHAPE[0])                                                  
            new_current_states = new_current_states.reshape(self.hp.MINIBATCH_SIZE, 
                                                            self.env.ENVIRONMENT_SHAPE[0])

        return current_states, new_current_states
    
    def compute_implied_policy(self, update):
        '''
        Function to compute the implied policy learned by the DQ learner. 
        ...
        Args
        ------
        self: class DrugSelector
        update: bool
            should we update the list of implied policies? 

        Returns numeric matrix 
            numeric matrix encodes policy in the same way as compute_optimal__policy
        '''
        policy = []

        if self.env.TRAIN_INPUT == "state_vector":
        
            for s in range(len(self.env.state_vector)):

                self.env.state_vector = np.zeros((2 ** self.env.N, 1))
                self.env.state_vector[s] = 1
                action = np.argmax(self.get_qs())
                policy.append(to_categorical(action, 
                            num_classes = len(self.env.drugs)))
                
        else:  #if the train input was fitness
            #put together action list
            a_list = to_categorical([i for i in range(len(self.env.ACTIONS))])
            a_list = np.ndarray.tolist(a_list)
            for s in range(len(self.env.state_vector)):
                state_vector = np.zeros((2 ** self.env.N, 1))
                state_vector[s] = 1
                a_out = []
                for a in range(len(a_list)):
                    fit = np.dot(self.env.drugs[a], state_vector)[0] #compute fitness for given state_vector, drug combination
                    a_vec = deepcopy(a_list)[a]
                    #append fitness to one-hot encoded action to mimic how the data are fed into the model
                    a_vec.append(fit)
                    a_vec = np.array(a_vec)
                    #reshape to feed into the model
                    tens = a_vec.reshape(-1, *self.env.ENVIRONMENT_SHAPE)
                    #find the optimal action
                    action_a = self.model.predict(tens)[0].argmax()
            
                    a_out.append(action_a)
                    
                policy.append(a_out)
            #policy_a = policy_a/len(a_list)
        
        if update:
            self.policies.append([policy, self.env.episode_number])
        else: #only return the policy if we are not updating anything
            return policy

    #function to get q vector for a given state
    def get_qs(self):
        if self.hp.TRAIN_INPUT == "state_vector":
            tens = self.env.state_vector.reshape(-1, *self.env.ENVIRONMENT_SHAPE)
        elif self.hp.TRAIN_INPUT == "fitness":
            #convert all
            sensor = self.env.sensor[3]
            tens = sensor.reshape(-1, *self.env.ENVIRONMENT_SHAPE)
        elif self.hp.TRAIN_INPUT == "pop_size":
            tens = self.env.pop_size.reshape(-1, *self.env.ENVIRONMENT_SHAPE)
        else: 
            return "error in get_qs()"

        return self.model.predict(tens)[0]
    
def compute_optimal_policy(agent, discount_rate = 0.99, num_steps = 20):
    '''
    Function to compute optimal policy based on reinforcement learning problem defined by the class DrugSelector
    ...
    Args
    ------
    agent: class DrugSelector 

    Returns numeric matrix 
        encoding optimal actions a for all states s in S
    '''

    env = dp_env(N = agent.env.N, sigma = agent.env.sigma, 
                 drugs = agent.env.drugs, num_drugs= len(agent.env.drugs))
    
    policy, V = backwards_induction(env = env, discount_rate= discount_rate, num_steps=num_steps)

    return policy

def compute_optimal_action(agent, policy, step, prev_action = False):
    '''
    Function to compute the optimal action based on a deterministic policy. 
    ...
    Args
    ------
    agent: class DrugSelector
    policy: numeric matrix 
        encoding optimal actions a for all states s in S

    Returns int 
        corresponding to optimal action
    '''
    
    index = [i for i,j in enumerate(agent.env.state_vector) if j == 1.][0]

    if prev_action:
        action = policy[index][int(agent.env.prev_action)] +1
    else:
        action = policy[index][step] + 1 #plus one because I made the bad decision to force the actions to be 1,2,3,4 once upon a time
    
    return action
    
#'main' function that iterates through simulations to train the agent
def practice(agent, naive = False, standard_practice = False, 
             dp_solution = False, pre_trained = False, discount_rate = 0.99,
             policy = "none", prev_action = False):
    '''
    Function that iterates through simulations to train the agent. Also used to test general drug cycling policies as controls for evodm 
    ...
    Args
    ------
    agent: class DrugSelector
    naive: bool
        should a naive drug cycling policy be used
    standard_practice: bool
        should a drug cycling policy approximating standard clinical practice be tested
    dp_solution: bool
        should a gold-standard optimal policy computed using backwards induction of an MDP be tested
    pre_trained: bool
        is the provided agent pre-trained? (i.e. should we be updating weights and biases each time step)
    prev_action: bool
        are we evaluating implied policies or actual DP policies?
    discount_rate: float
    policy: numeric matrix 
        encoding optimal actions a for all states s in S, defaults to "none" - 
        in which case logic defined by bools will dictate which policy is used. 
        If a policy is provided, it will supercede all other options and be tested

    Returns rewards, agent, policy 
        reward vector, trained agent including master memory dictating what happened, and learned policy (if applicable)
    '''
    if dp_solution:
        dp_policy = compute_optimal_policy(agent, discount_rate = discount_rate,
                                          num_steps = agent.hp.RESET_EVERY)

    #this is a bit of a hack - we are coopting the code that tests the dp solution to
    #  test user-provided policies that use the same format
    #These policies will almost never have anything to do with the dp solutions
    if policy != "none": 
        dp_policy = policy
        dp_solution = True

    #every given number of episodes we are going to track the stats
    #format is [average_reward, min_reward, max_reward]
    reward_list = []
    #initialize list of per episode rewards
    ep_rewards = []
    for episode in tqdm(range(1, agent.hp.EPISODES + 1), ascii=True, unit='episodes', 
                        disable = True if dp_solution else False):

        # Restarting episode - reset episode reward and step number
        episode_reward = 0
        if pre_trained:
            agent.hp.epsilon = 0

        for i in range(agent.hp.RESET_EVERY):

            # This part stays mostly the same, the change is to query a model for Q values
            if np.random.random() > agent.hp.epsilon:
                # Get action from Q table
                if naive:
                    if standard_practice:
                        #Only change the action if fitness is above 0.9
                        if np.mean(agent.env.fitness) > 0.9:
                            avail_actions = [action for action in agent.env.ACTIONS if action != agent.env.action] #grab all actions except the one currently selected
                            agent.env.action = random.sample(avail_actions, k = 1)[0] #need to take the first element of the list because thats how random.sample outputs it
                    else: 
                        agent.env.action = random.randint(np.min(agent.env.ACTIONS),np.max(agent.env.ACTIONS))
                elif dp_solution:
                    agent.env.action = compute_optimal_action(agent, dp_policy, step = i, prev_action=prev_action)
                else:
                    agent.env.action = np.argmax(agent.get_qs()) + 1 #plus one because of the stupid fucking indexing system
            else:
                # Get random action
                if standard_practice:
                        #Only change the action if fitness is above 0.9
                    if np.mean(agent.env.fitness) > 0.9:
                        avail_actions = [action for action in agent.env.ACTIONS if action != agent.env.action] #grab all actions except the one currently selected
                        agent.env.action = random.sample(avail_actions, k = 1)[0] #need to take the first element of the list because thats how random.sample outputs it
                elif dp_solution:
                    agent.env.action = compute_optimal_action(agent, dp_policy, step = i, prev_action = prev_action)
                else: 
                    agent.env.action = random.randint(np.min(agent.env.ACTIONS),np.max(agent.env.ACTIONS))


            #we don't save anything - it stays in the class
            agent.env.step()

            # Transform new continous state to new discrete state and count reward
            # can't do this after just 1 step because there won't be anything in the sensor
            if i != 0:
                reward = agent.env.sensor[2]
                episode_reward += reward

                # Every step we update replay memory and train main network - only train if we are doing a not naive run
                agent.update_replay_memory()
                if not any([dp_solution, naive, pre_trained]):
                    agent.train()
            
            if agent.env.done: # break if either of the victory conditions are met
                break #check out calc_reward in the evol_env class for how this is defined

        # Append episode reward to a list and log stats (every given number of episodes)
        ep_rewards.append(episode_reward)
        if not episode % agent.hp.AGGREGATE_STATS_EVERY or episode == 1:
            average_reward = sum(
                ep_rewards[-agent.hp.AGGREGATE_STATS_EVERY:])/len(ep_rewards[-agent.hp.AGGREGATE_STATS_EVERY:])
            min_reward = min(ep_rewards[-agent.hp.AGGREGATE_STATS_EVERY:])
            max_reward = max(ep_rewards[-agent.hp.AGGREGATE_STATS_EVERY:])
            reward_list.append([episode, average_reward, min_reward, max_reward])

            #update the implied policy vector
            if not any([dp_solution, naive, pre_trained]):
                if not agent.hp.NUM_EVOLS > 1:
                    agent.compute_implied_policy(update = True)

            # Save model, but only when min reward is greater or equal a set value
            # haven't figured out what min reward is for that
            #if min_reward >= MIN_REWARD:
             #   agent.model.save(
              #      f'models/{MODEL_NAME}__{max_reward:_>7.2f}max_{average_reward:_>7.2f}avg_{min_reward:_>7.2f}min__{int(time.time())}.model')

        # Decay epsilon - only if agent is not naive -- since we are calling it twice
        if not naive:
            if agent.hp.epsilon > agent.hp.MIN_EPSILON:
                agent.hp.epsilon *= agent.hp.EPSILON_DECAY
                agent.hp.epsilon = max(agent.hp.MIN_EPSILON, agent.hp.epsilon)

        # reset environment for next iteration
        agent.env.reset()
    if dp_solution:
        policy = dp_policy
    elif naive:
        policy = []
    elif pre_trained:
        policy = []
    else:
        policy = agent.compute_implied_policy(update = False)
    return reward_list, agent, policy

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
        rewards_i, agent_i, policy_i = practice(deepcopy(agent), dp_solution = True, discount_rate = i)
        mem_i = agent_i.master_memory
        mem_list.append([mem_i, i])
        policy_list.append([policy_i, i])

    return [mem_list, policy_list]

def test_generic_policy(policy, episodes = 100, num_steps = 20, normalize_drugs= False, 
                        prev_action = False):
    '''
    Function to test a generic policy for performance in simulated e.coli system
    Args:
        policy: list of lists
            policy matrix specifying deterministic action selection for each state-evolstep. 
        episodes: int
        num_steps: int
        normalize_drugs= bool
    
    returns list of lists
        agent memory specifying performance.
    '''
    hp = hyperparameters()
    hp.EPISODES = episodes
    hp.RESET_EVERY = num_steps
    hp.N = 4
    hp.NUM_DRUGS = 15
    hp.NORMALIZE_DRUGS = normalize_drugs
    
    drugs = define_mira_landscapes()
    agent = DrugSelector(hp = hp, drugs = drugs)

    rewards,agent, policy = practice(deepcopy(agent), dp_solution=True, 
                                     policy = policy, prev_action = prev_action)

    mem = agent.master_memory
    return mem

def sweep_replicate_policy(agent, episodes = 500):
    '''
    Function to sweep the policy learned by a given replicate at every episode
    Args:
        episodes: int
            how many episodes should be evaluated per policy
        normalize_drugs: bool
    '''    
    policies = agent.policies
    reset = agent.hp.RESET_EVERY

    mem_list = []
    for i in range(len(policies)): 
        policy = policies[i][0]
        mem_i = test_generic_policy(policy, num_steps = reset, 
                                    prev_action = True, episodes=episodes)
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
            rewards_i, agent_i, policy = practice(deepcopy(agent), dp_solution = True, policy=policy_i)
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
    
    
def evol_deepmind(num_evols = 1, N = 5, episodes = 50,
                  reset_every = 20, min_epsilon = 0.005, 
                  train_input = "fitness",  random_start = False, 
                  noise = False, noise_modifier = 1, num_drugs = 4, 
                  sigma = 0.5, normalize_drugs = True, 
                  player_wcutoff = 0.1, pop_wcutoff = 0.99, win_threshold = 200,
                  win_reward = 0, standard_practice = False, drugs = "none",
                  average_outcomes = False, mira = False, gamma = 0.99,
                  learning_rate = 0.002, minibatch_size = 400, 
                  pre_trained = False, 
                  agent = "none",
                  update_target_every = 310):
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

    #gotta modulate epsilon decay based on the number of episodes defined
    #0.005 = epsilon_decay^episodes
    hp.EPSILON_DECAY = pow(hp.MIN_EPSILON, 1/hp.EPISODES)

    if pre_trained and agent != "none":
        agent.master_memory = []
        rewards, agent, policy = practice(agent, naive=False, pre_trained = pre_trained)
        return [rewards, agent, policy]
        
    if mira:
        hp.NORMALIZE_DRUGS = False
        drugs = define_mira_landscapes()
    #initialize agent, including the updated hyperparameters
    agent = DrugSelector(hp = hp, drugs = drugs)
    naive_agent = deepcopy(agent) #otherwise it all gets overwritten by the actual agent
    if not average_outcomes:
        dp_agent = deepcopy(agent)
        dp_rewards, dp_agent, dp_policy = practice(dp_agent, dp_solution = True, 
                                                   discount_rate= hp.DISCOUNT)

    #run the agent in the naive case and then in the reg case
    naive_rewards, naive_agent, naive_policy = practice(naive_agent, naive = True, standard_practice=standard_practice)
    rewards, agent, policy = practice(agent, naive = False)


    return [rewards, naive_rewards, agent, naive_agent, dp_agent, dp_rewards,
            dp_policy, naive_policy, policy]

#rewards = evol_deepmind()
#naive_rewards= evol_deepmind(naive = True)
