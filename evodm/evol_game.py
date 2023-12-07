from evodm.landscapes import Landscape
import numpy as np
from tensorflow.keras.utils import to_categorical
import math
import random
import itertools
import copy
# Functions to convert data describing bacterial evolution sim into a format
# that can be used by the learner

#
# The environment.step method takes an action in the environment and returns a
# TimeStep tuple containing the next observation of the environment and the reward for the action.
#

class evol_env:

    #intialize the environment class - key variable is "train_input"
    #train input determines whether the sensor records the state vector or
    #the previous fitnesses
    def __init__(self,  drugs = None, N = 5, sigma = 0.5, correl = np.linspace(-1.0,1.0,51),
                        phenom = 0,
                        train_input = "state_vector", num_evols = 1,
                        random_start = False, 
                        num_drugs = 4, 
                        normalize_drugs = True, 
                        win_threshold = 10,
                        player_wcutoff = 0.8, 
                        pop_wcutoff = 0.95, 
                        win_reward = 10, 
                        noise_modifier = 1, 
                        add_noise = True, 
                        average_outcomes = False, 
                        total_resistance = False,
                        starting_genotype = 0, 
                        dense= False, cs = False, 
                        delay = 0):
        #define switch for whether to record the state vector or fitness for the learner
        self.TRAIN_INPUT = train_input
        #define environmental variables
        self.time_step = 0 #define for number of evolutionary steps counter
        self.action_number = 0 #define number of actions being taken
        self.episode_number = 1 #count the number of times through a given fitness paradigm we are going
        if train_input == "pop_size" and num_evols !=1: 
            print("consider setting num_evols to 1 when using population size as the train input")
        self.NUM_EVOLS = num_evols #number of evolutionary steps per sample
        self.NUM_OBS = 100 #number of "OD" readings per evolutionary step 
        self.pop_size = []  
        #define actions  
        self.ACTIONS = [i for i in range(1, num_drugs + 1)] # action space - added the plus one because I decided to use 1 indexing for the actions for no good reason a while ago
        self.action = 1 #first action - value will be updated by the learner
        self.prev_action = 1.0 #pretend this is the second time seeing it why not
        self.update_target_counter = 0
        

        #should noise be introduced into the fitness readings?
        self.NOISE_MODIFIER = noise_modifier
        self.NOISE_BOOL = add_noise
        self.AVERAGE_OUTCOMES = average_outcomes

        #measurement delay (i.e. are state vector readings delayed by n time steps)
        self.DELAY = delay
        if self.DELAY > 0:
            self.state_mem = []
        #data structure for containing information the agent queries from the environment.
        # the format is: [previous_fitness, current_action, reward, next_fitness]
        self.sensor =  []
        #save environment variables
        self.N = N
        self.sigma = sigma
        self.DENSE= dense
        self.CS = cs
        self.num_drugs = num_drugs
        self.PHENOM = phenom
       
        #Defining these in case self.reset is called
        self.correl = correl

        #define victory counters
        self.player_wcount = 0
        self.pop_wcount = 0

        #define victory conditions for player and pop
        self.PLAYER_WCUTOFF = player_wcutoff
        self.POP_WCUTOFF = pop_wcutoff

        #define victory threshold
        self.WIN_THRESHOLD = win_threshold # number of player actions before the game is called
        self.WIN_REWARD = win_reward
        self.STARTING_GENOTYPE = starting_genotype

        self.done = False

        #should each episode start with a random scatter of genotypes?
        self.RANDOM_START = random_start
        self.TOTAL_RESISTANCE = total_resistance

        self.define_landscapes(drugs=drugs, normalize_drugs=normalize_drugs)

        ##initialize state vector
        if self.RANDOM_START:
            self.state_vector = np.ones((2**N,1))/2**N
        else:
            self.state_vector  = np.zeros((2**N,1))
            self.state_vector[starting_genotype][0] = 1

        self.update_state_mem(state_vector = self.state_vector)

        ##Define initial fitness
        self.fitness = [np.dot(self.drugs[self.action-1], self.state_vector)]
        if self.NOISE_BOOL:
            self.sensor_fitness = self.add_noise(self.fitness)
        else:
            self.sensor_fitness = self.fitness
        
        #Define the environment shape
        if self.TRAIN_INPUT == "state_vector":#give the state vector for every evolution
            self.ENVIRONMENT_SHAPE = (len(self.state_vector),1)
        elif self.TRAIN_INPUT == "fitness":
            self.ENVIRONMENT_SHAPE = (num_drugs + self.NUM_EVOLS,)
        elif self.TRAIN_INPUT == "pop_size":
            self.ENVIRONMENT_SHAPE = (self.NUM_OBS, 1)
        else:
                print("please specify state_vector, pop_size, or fitness for train_input when initializing the environment")
                return

    def define_landscapes(self, drugs, normalize_drugs):
        #default behavior is to generate landscapes completely at random. 
        #define landscapes #this step is a beast - let's pull this out into it's own function
        if drugs is None: 
            ## Generate landscapes - use whatever parameters were set in main()
            self.landscapes, self.drugs = generate_landscapes(N = self.N, 
                                                  sigma = self.sigma,
                                                  num_drugs = self.num_drugs,
                                                  correl=self.correl, dense = self.DENSE,
                                                  CS=self.CS)
            
        else:
            self.drugs = drugs #use pre-defined drugs 
            self.landscapes = [Landscape(ls = i, N=self.N, sigma = self.sigma, 
                                         dense = self.DENSE) for i in self.drugs]

        #Normalize landscapes if directed
        if normalize_drugs:
            self.drugs = normalize_landscapes(self.drugs)
            self.landscapes = [Landscape(ls = i, N=self.N, sigma = self.sigma,
                                         dense = self.DENSE) for i in self.drugs]
        #if self.PHENOM > 0:
        #    [i.get_TM_phenom(phenom = self.PHENOM) for i in self.landscapes]
        #else:
        #    [i.get_TM() for i in self.landscapes] #pre-compute TM
        [i.get_TM_phenom(phenom = self.PHENOM) for i in self.landscapes]
        
        return 


    def step(self):

        #update current time step of the sim
        #update how many actions have been taken
        self.time_step += self.NUM_EVOLS
        self.action_number += 1
        self.update_target_counter +=1
        
        # Run the sim under the assigned conditions
        if self.action not in self.ACTIONS:
            return("the action must be in env.ACTIONS")
        fitness, state_vector = run_sim(evol_steps = self.NUM_EVOLS,
                                        state_vector = self.state_vector,
                                        ls = self.landscapes[self.action-1], 
                                        average_outcomes=self.AVERAGE_OUTCOMES)
        
        
        self.update_state_mem(state_vector = state_vector)

        if self.NOISE_BOOL:
            sensor_fitness = self.add_noise(fitness)
        else:
            sensor_fitness = fitness

        self.define_sensor(fitness=fitness, sensor_fitness=sensor_fitness, 
                           state_vector=state_vector)
        
        #update vcount
        self.update_vcount(fitness = fitness)

        #update the current fitness vector
        self.fitness = fitness
        self.sensor_fitness = sensor_fitness
        #update the current state vector
        self.state_vector = state_vector
        #update action-1 - its assumed that self.action is updated prior to initiating env.step
        self.prev_action = float(self.action) #type conversion

        #done
        return
    
    def update_state_mem(self, state_vector):
        if self.DELAY >0:
            if self.TRAIN_INPUT == "fitness":
                return("measurement delay only supported when train input is state_vector")
            self.state_mem.append(state_vector)
        return
    
    def define_sensor(self, fitness, sensor_fitness, state_vector):
        # Again, this is creating a stacked data structure where each time point provides
        # [current_state, current_action, reward, next_state]
        # [s_t, a_t, r_t, s_t+1]
        # if self.delay > 0, this list will look like:
        # [s_{t-delay}, a_t, r_t, s_{t-delay+1}]

        if self.DELAY > 0 and len(self.state_mem) <= self.DELAY+1:
            return              
        if self.TRAIN_INPUT == "state_vector":
            if self.DELAY > 0:
                index = self.time_step-1 - self.DELAY #because time step is 1 indexed
                state1 = self.state_mem[index] 
                state2 = self.state_mem[index + 1]
            else:
                state1 = self.state_vector 
                state2 = state_vector 
            
            self.sensor = [state1, self.action, self.calc_reward(fitness = fitness), state2]
        elif self.TRAIN_INPUT == "fitness":
            if self.NUM_EVOLS > 1 and self.time_step ==self.NUM_EVOLS: #otherwise the prev_fitness and fitness objects won't share the same shape.
                return
            #convert fitness + action into trainable state vector for n and n+1
            prev_action_cat, action_cat = self.convert_fitness(fitness = sensor_fitness)
            self.sensor= [prev_action_cat, 
                            self.action, self.calc_reward(fitness = fitness), 
                            action_cat] 
        elif self.TRAIN_INPUT == "pop_size":
            ##Here we interpolate between the previous fitness and the next fitness
            pop_size = self.growth_curve(new_fitness = fitness)
            self.sensor= [self.pop_size, self.action, self.calc_reward(fitness = fitness), pop_size]
                #update pop size vector
            self.pop_size = pop_size    
        else:
            print("please specify either state_vector, fitness, or pop_size for train_input when initializing the environment")
            return

    def convert_fitness(self, fitness): 
        #convert to lists
        if self.NUM_EVOLS > 1:
            prev_fitness = np.ndarray.tolist(self.sensor_fitness)
            fitness = np.ndarray.tolist(fitness)
        else: 
            prev_fitness = self.fitness

        prev_action_cat = to_categorical(self.prev_action-1, num_classes = len(self.ACTIONS)) #-1 because of the dumb python indexing system
        prev_action_cat = np.ndarray.tolist(prev_action_cat) #convert to list

        #This checks if fitness is a list (will occur if num_evols > 1)
        if isinstance(prev_fitness, list):
        #append fitness values
            for i in range(len(prev_fitness)):
                prev_action_cat.append(prev_fitness[i])
        else:
            prev_action_cat.append(prev_fitness)

        action_cat = to_categorical(self.action-1, num_classes = len(self.ACTIONS))
        action_cat = np.ndarray.tolist(action_cat)
         #This checks if fitness is a list (will occur if num_evols > 1)
        if isinstance(fitness, list):  
            for i in range(len(fitness)):
                action_cat.append(fitness[i])  
        else:
            action_cat.append(fitness)

        return prev_action_cat, action_cat

    def update_vcount(self, fitness):
        if np.mean(fitness) < self.PLAYER_WCUTOFF: 
            self.player_wcount += 1
        else: 
            self.player_wcount = 0 #make sure self.player_wcount only climbs with consecutive scores in that territory
        
        if np.mean(fitness) > self.POP_WCUTOFF:
            self.pop_wcount += 1
        else:
            self.pop_wcount = 0
        return

    def compute_average_fitness(self):
        #function to compute the average fitness to all available drugs in the panel
        fitnesses = []
        for i in iter(self.ACTIONS):
            fitness = np.dot(self.drugs[i-1], self.state_vector)
            fitnesses.append(fitness)

        return np.mean(fitnesses)

    def calc_reward(self, fitness, total_resistance = False):
    #okay so if fitness is low - this number will be higher
    #"winning" the game is associated with a huge reward while losing a huge penalty

        if total_resistance:
            if self.pop_wcount >= self.WIN_THRESHOLD:
                reward = -self.WIN_REWARD
                self.DONE = True
            elif self.player_wcount >= self.WIN_THRESHOLD: 
                reward = self.WIN_REWARD
                self.DONE = True
            else: 
                #need to compute the average fitness across all drugs and then do 1- that
                reward = (1 - self.compute_average_fitness)
        else:
            if self.pop_wcount >= self.WIN_THRESHOLD:
                reward = -self.WIN_REWARD
                self.DONE = True
            elif self.player_wcount >= self.WIN_THRESHOLD: 
                reward = self.WIN_REWARD
                self.DONE = True
            else: 
                reward = np.mean(1 - fitness)

        return reward
    
    def add_noise(self, fitness):
        noise_param = np.random.normal(loc = 0, 
                                       scale = (0.05 * self.NOISE_MODIFIER))

        #muddy the fitness value with noise
        if type(fitness) is not list: 
            return fitness + noise_param
        else:
            return [i + noise_param for i in fitness]

    def growth_curve(self, new_fitness):
        new_fitness = np.mean(new_fitness)
        old_fitness = np.mean(self.fitness)
        
        #handle edge cases where fitness is 0 or 1 (not sure its even possible)
        if old_fitness == 1:
            old_fitness = 0.999
        elif old_fitness == 0:
            old_fitness = 0.001
        
        if new_fitness == 1:
            new_fitness = 0.999
        elif new_fitness == 0:
            new_fitness = 0.001
        
        #generate untransformed od_dist
        od_dist_raw = np.linspace(s_solve(old_fitness), s_solve(new_fitness), 
                                  num = self.NUM_OBS)

        state_vector = [1/(1+math.exp(-i)) for i in od_dist_raw] #generate bounded sigmoid function

        return np.array(state_vector)
    

    def reset(self):
        ##re-initialize state vector
        ##initialize state vector
        if self.DELAY > 0:
            self.state_mem = []

        if self.RANDOM_START:
            self.state_vector = np.ones((2**self.N,1))/2**self.N
        else:
            self.state_vector  = np.zeros((2**self.N,1))
            self.state_vector[self.STARTING_GENOTYPE][0] = 1

        #reset time step
        self.time_step=0

        #reset fitness vector and action number
        self.fitness = []
        self.action_number = 0

        #advance episode number
        self.episode_number += 1

        #re-initialize the action number
        self.action = 1

        #re-initialize victory conditions
        self.pop_wcount = 0
        self.player_wcount = 0
        self.done = False
        #re-calculate fitness with the new state_vector
        self.fitness = [np.dot(self.drugs[self.action-1], self.state_vector)]
        if self.NOISE_BOOL: 
            self.fitness = self.add_noise(self.fitness)
        self.sensor = []

#helper function for generating the od_dist
def s_solve(y):
        x = -math.log(1/y - 1)
        return x

#additional methods used by prep_environ. too lazy to make them part of the class
def generate_landscapes(N = 5, sigma = 0.5, correl = np.linspace(-1.0,1.0,51), 
                        dense = False, CS = False, num_drugs = 4):

    A = Landscape(N, sigma, dense = dense)
    #give it two chances at this step because sometimes it doesn't converge
    try:
        Bs = A.generate_correlated_landscapes(correl)
    except:
        Bs = A.generate_correlated_landscapes(correl)

    if CS: 
        #this code guarantees that high-level CS will be present 
        split_index = np.array_split(range(len(Bs)), num_drugs)
        keep_index = [round(np.median(i)) for i in split_index]
    else:
        keep_index = np.random.randint(0, len(Bs)-1, size = num_drugs)
    
    landscapes = [Bs[i] for i in keep_index]
    drugs = [i.ls for i in landscapes]

    return landscapes, drugs

    #pprint.pprint(drugs)           output all the drugs
    #pprint.pprint(drugs[3,0])      output the fitness of genotype 4 in drug 1.
def generate_landscapes2(N=4, sigma=0.5, num_drugs=4, CS = False, dense = False, correl = None):
    #this in theory should be much cheaper than generate_landscapes which ensures 
    # that there are a range of correlations between the returned landscapes
    landscapes = []
    for i in range(num_drugs):
        landscapes.append(Landscape(N, sigma))

    drugs = [i.ls for i in landscapes]
    return landscapes, drugs

def normalize_landscapes(drugs):
    drugs_normalized = []
    for i in range(len(drugs)): 
        # this should transform everthing so the fitness range is 0-1 for all landscapes
        drugs_i = drugs[i] - np.min(drugs[i])
        drugs_normalized.append(drugs_i / np.max(drugs_i))
        
    return drugs_normalized
#function to progress the sim by n steps (using the evol_steps parameter)
# Naive is a logical flag - set true to run the simulation in a naive case 
# with random drug switching every 5th cycle 
def discretize_state(state_vector):
    '''
    Helper Function to discretize state vector - 
    converting the returned average outcomes to a single population trajectory.
    '''
    S = [i for i in range(len(state_vector))]
    probs = state_vector.reshape(len(state_vector))
    #choose one state - using the relative frequencies of the other states as the probabilities of being selected
    state = np.random.choice(S, size = 1, p = probs) #pick a state for the whole p
    new_states = np.zeros((len(state_vector),1))
    new_states[state] = 1
    return new_states


def run_sim(evol_steps, ls, state_vector, average_outcomes = False):
    '''
    Function to progress evolutionary simulation forward n times steps in a given fitness regime defined by action

    Args
        evol_steps: int
            number of steps
        state_vector: array
            N**2 length array defining the position of the population in genotype space
        average_outcomes bool
            should all possible futures be averaged into the state vector or should 
            we simulate a single evolutionary trajectory? defaults to False
    Returns: fitness, state_vector
        fitness: 
            population fitness in chosen drug regime
    '''
    reward = []
    # Evolve for 100 steps.
    for i in range(evol_steps):
        # This is the fitness of the population when the drug is selected to be used.
        if not average_outcomes: 
            state_vector = discretize_state(state_vector)

        reward.append(np.dot(ls.ls,state_vector))  

        # Performs a single evolution step - TM should be stored in the landscape object
        state_vector = ls.evolve(1, p0=state_vector)
        
    if not average_outcomes:
        state_vector = discretize_state(state_vector) #discretize again before sending it back
        
    reward = np.squeeze(reward)
    return reward, state_vector

#supposedly faster than numpy.random.choice
def fast_choice(options, probs):
    x = random.random()
    cum = 0
    for i, p in enumerate(probs):
        cum += p
        if x < cum:
            return options[i]
    return options[-1]

def define_mira_landscapes(as_dict = False):
    '''
    Function to define the landscapes described in 
    Mira PM, Crona K, Greene D, Meza JC, Sturmfels B, Barlow M (2015) 
    Rational Design of Antibiotic Treatment Plans: A Treatment Strategy for Managing Evolution and Reversing Resistance. 
    PLoS ONE 10(5): e0122283. https://doi.org/10.1371/journal.pone.0122283
    '''
    if as_dict:
        drugs = {}
        drugs['AMP'] = [1.851, 2.082, 1.948, 2.434, 2.024, 2.198, 2.033, 0.034, 1.57, 2.165, 0.051, 0.083, 2.186, 2.322, 0.088, 2.821]    #AMP
        drugs['AM'] = [1.778, 1.782, 2.042, 1.752, 1.448, 1.544, 1.184, 0.063, 1.72, 2.008, 1.799, 2.005, 1.557, 2.247, 1.768, 2.047]    #AM
        drugs['CEC'] = [2.258, 1.996, 2.151, 2.648, 2.396, 1.846, 2.23, 0.214, 0.234, 0.172, 2.242, 0.093, 2.15, 0.095, 2.64, 0.516]      #CEC
        drugs['CTX'] = [0.16, 0.085, 1.936, 2.348, 1.653, 0.138, 2.295, 2.269, 0.185, 0.14, 1.969, 0.203, 0.225, 0.092, 0.119, 2.412]     #CTX
        drugs['ZOX'] = [0.993, 0.5, 2.069, 2.683, 1.698, 2.01, 2.138, 2.688, 1.106, 1.171, 1.894, 0.681, 1.116, 1.105, 1.103, 2.591]   #ZOX
        drugs['CXM'] = [1.748, 1.7, 2.07, 1.938, 2.94, 2.173, 2.918, 3.272, 0.423, 1.578, 1.911, 2.754, 2.024, 1.678, 1.591, 2.923]      #CXM
        drugs['CRO'] =[1.092, 0.287, 2.554, 3.042, 2.88, 0.656, 2.732, 0.436, 0.83, 0.54, 3.173, 1.153, 1.407, 0.751, 2.74, 3.227]       #CRO
        drugs['AMC'] =[1.435, 1.573, 1.061, 1.457, 1.672, 1.625, 0.073, 0.068, 1.417, 1.351, 1.538, 1.59, 1.377, 1.914, 1.307, 1.728]    #AMC
        drugs['CAZ'] =[2.134, 2.656, 2.618, 2.688, 2.042, 2.756, 2.924, 0.251, 0.288, 0.576, 1.604, 1.378, 2.63, 2.677, 2.893, 2.563]    #CAZ
        drugs['CTT'] =[2.125, 1.922, 2.804, 0.588, 3.291, 2.888, 3.082, 3.508, 3.238, 2.966, 2.883, 0.89, 0.546, 3.181, 3.193, 2.543]    #CTT
        drugs['SAM'] =[1.879, 2.533, 0.133, 0.094, 2.456, 2.437, 0.083, 0.094, 2.198, 2.57, 2.308, 2.886, 2.504, 3.002, 2.528, 3.453]    #SAM
        drugs['CPR'] =[1.743, 1.662, 1.763, 1.785, 2.018, 2.05, 2.042, 0.218, 1.553, 0.256, 0.165, 0.221, 0.223, 0.239, 1.811, 0.288]    #CPR
        drugs['CPD'] =[0.595, 0.245, 2.604, 3.043, 1.761, 1.471, 2.91, 3.096, 0.432, 0.388, 2.651, 1.103, 0.638, 0.986, 0.963, 3.268]    #CPD
        drugs['TZP'] =[2.679, 2.906, 2.427, 0.141, 3.038, 3.309, 2.528, 0.143, 2.709, 2.5, 0.172, 0.093, 2.453, 2.739, 0.609, 0.171]     #TZP
        drugs['FEP'] =[2.59, 2.572, 2.393, 2.832, 2.44, 2.808, 2.652, 0.611, 2.067, 2.446, 2.957, 2.633, 2.735, 2.863, 2.796, 3.203]     #FEP
    else:
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

#Function to compute reward for a given simulation step - used by the environment class. 
#Could have defined this in-line but made it a separate function in case we want to make it 
#more sophisticated in the future. 

class evol_env_wf:
    def __init__(self, N= 4, num_drugs = 15, train_input = 'state_vector', pop_size = 10000, 
                 gen_per_step = 2, mutation_rate = 1e-6, hgt_rate = 1e-5, 
                 drugLandscape = define_mira_landscapes()):

        self.update_target_counter= 0 
        #save everything
        self.episode_number = 1
        self.N= N
        self.pop_size = pop_size
        self.gen_per_step = gen_per_step
        self.mutation_rate = mutation_rate
        self.hgt_rate = hgt_rate
        self.TRAIN_INPUT = train_input
        self.NUM_DRUGS = num_drugs
        self.pop = {}
        self.sensor = []
        self.history = []
        self.fit = {}

        self.alphabet = ['0', '1']
        self.base_haplotype = ''.join(["0" for i in range(self.N)])

        genotypes = [''.join(seq) for seq in itertools.product("01", repeat=self.N)]
        drugs = []

        self.pop[self.base_haplotype] = self.pop_size

        for drug in range(self.NUM_DRUGS):
            for i in range(len(genotypes)):
                self.fit[genotypes[i]] = drugLandscape[drug][i]

            drugs.append(copy.deepcopy(self.fit))
            self.fit.clear()

        self.drugs = drugs

        self.action = 0
        #select the first drug
        self.drug = self.drugs[self.action]
        self.prev_drug=self.drug
        self.prev_action = 0.0

        #housekeeping
        self.step_number = 1 #step_number is analagous to step in the original simulations
        self.time_step_number=0 #time step number is the generation count
        self.fitness = self.compute_pop_fitness(drug = self.drug, sv = self.pop)
        self.state_vector = self.convert_state_vector(sv = self.pop)
        #Stuff to make things not break
        if self.TRAIN_INPUT == "state_vector":#give the state vector for every evolution
            self.ENVIRONMENT_SHAPE = (len(self.state_vector),1)
        elif self.TRAIN_INPUT == "fitness":
            self.ENVIRONMENT_SHAPE = (self.NUM_DRUGS + 1,)

        self.action_number = 0

        self.ACTIONS = [i for i in range(self.NUM_DRUGS)] # action space -no plus one because it was really really dumb 

        self.done = False
  
    def update_sensor(self, pop):
        # this is creating a stacked data structure where each time point provides
        # [current_fitness, current_action, reward, next_fitness]
        #or [current_state, current_action, reward, next_state]
        if self.TRAIN_INPUT == "state_vector":
            sv = self.convert_state_vector(sv = pop)
            sv_prime = self.convert_state_vector(sv = self.pop) #self.pop is 
        elif self.TRAIN_INPUT == 'fitness':
            fit = self.compute_pop_fitness(sv = pop, drug = self.prev_drug)
            fit_prime = self.compute_pop_fitness(sv=self.pop, drug = self.drug)
            sv, sv_prime = self.convert_fitness(fitness = fit,fitness_prime = fit_prime)
        
        fit = self.compute_pop_fitness(sv=self.pop, drug = self.drug)
        self.sensor = [sv, self.action, 1-fit, sv_prime] #reward is just 1-fitness
    
    def compute_pop_fitness(self, drug, sv):
        x = [(sv[i] * drug[i]) / self.pop_size for i in sv.keys()]
        fit = np.sum(x)
        return fit
    
    def convert_state_vector(self, sv):
        new_sv = np.zeros((self.N**2,1))
        keys = list(sv.keys())
        for i in range(len(sv)):
            #convert binary key to numeric
            state = int(keys[i], 2)
            val = sv[keys[i]] / self.pop_size
            new_sv[state][0] = val

        return new_sv
    def update_drug(self, drug_num):
        #Always use this method to update the drug
        self.action = drug_num
        self.prev_drug = self.drug
        self.drug = self.drugs[self.action]
        
    def step(self): #just renaming this to match with the base evol_env format
        #update action number
        self.update_target_counter += 1
        self.action_number += 1

        pop_old = dict(self.pop)
        if self.time_step_number == 0:
            self.history.append(pop_old)
        for i in range(self.gen_per_step):
            self.time_step()
            clone_pop = dict(self.pop)
            self.history.append(clone_pop)
            self.time_step_number += 1
        #prep for next 'step'
        self.time_step_number=1
        self.step_number +=1
        self.update_sensor(pop=pop_old)
        self.prev_action = float(self.action)
        self.state_vector = self.convert_state_vector(sv = self.pop)
        self.fitness = self.compute_pop_fitness(drug = self.drug, sv = self.pop)
        
    #reset the environment after an 'episode'
    def reset(self):
        self.episode_number +=1 
        self.action_number = 0
        self.time_step_number = 0
        self.step_number = 1
        self.pop = {}
        self.sensor = []
        self.pop[self.base_haplotype] = self.pop_size

        #reset drug stuff to baseline as well
        self.action = 0
        #select the first drug
        self.drug = self.drugs[self.action]
        self.state_vector = self.convert_state_vector(sv = self.pop)
        self.fitness = self.compute_pop_fitness(drug = self.drug, sv = self.pop)

    def time_step(self):
        self.mutation_step()
        self.hgt_event()
        self.offspring_step()

    def mutation_step(self):
        mutation_count = self.get_mutation_count()
        for i in range(mutation_count):
            self.mutation_event()

    def hgt_step(self):
        hgt_count = self.get_hgt_count()
        for i in range(hgt_count):
            self.hgt_event()

    def get_mutation_count(self):
        mean = self.mutation_rate * self.pop_size * self.N
        return np.random.poisson(mean)

    """
    Function that find a random haplotype to mutate and adds that new mutant to the population. Reduces mutated population by 1.
    """
    def mutation_event(self):
        haplotype = self.get_random_haplotype()
        if self.pop[haplotype] > 1:
            self.pop[haplotype] -= 1
            new_haplotype = self.get_mutant(haplotype)
            if new_haplotype in self.pop:
                self.pop[new_haplotype] += 1
            else:
                self.pop[new_haplotype] = 1

    """
    Function that gets the number of hgt events we should see.
    """
    def get_hgt_count(self):
        mean = self.hgt_rate * self.pop_size * self.N
        return np.random.poisson(mean)
    
    """
    Function that find a random pair of haplotypes to mutate and do the hgt event
    """
    def hgt_event(self):
        haplotype_1 = self.get_random_haplotype()
        haplotype_2 = self.get_random_haplotype()
        new_hap2 = ""
        for i in range(len(haplotype_1)):
            if haplotype_1[i] == '1' and haplotype_2[i] == '0':
                new_hap2 += "1"
            else:
                new_hap2 += haplotype_2[i]
        
        self.pop[haplotype_2] -=1

        if new_hap2 in self.pop:
            self.pop[new_hap2] += 1
        else: 
            self.pop[new_hap2] = 1

    """
    Chooses a random haplotype in the population that will be returned.
    """
    def get_random_haplotype(self):
        haplotypes = list(self.pop.keys())
        frequencies = [x/self.pop_size for x in self.pop.values()]
        total = sum(frequencies)
        frequencies = [x / total for x in frequencies]
        return fast_choice(haplotypes, frequencies)
        #return random.choices(haplotypes, weights=frequencies)[0]

    """
    Receives the haplotype to be mutated and returns a new haplotype with a mutation with all neighbor mutations equally probable.
    """
    def get_mutant(self, haplotype):
        site = int(random.random()*self.N)
        possible_mutations = list(self.alphabet)
        possible_mutations.remove(haplotype[site])
        mutation = random.choice(possible_mutations)
        new_haplotype = haplotype[:site] + mutation + haplotype[site+1:]
        return new_haplotype

    def convert_fitness(self, fitness, fitness_prime): 
        #convert to lists
        
        prev_action_cat = to_categorical(self.prev_action-1, num_classes = len(self.ACTIONS)) #-1 because of the dumb python indexing system
        prev_action_cat = np.ndarray.tolist(prev_action_cat) #convert to list

        #This checks if fitness is a list (will occur if num_evols > 1)
        
        prev_action_cat.append(fitness)

        action_cat = to_categorical(self.action-1, num_classes = len(self.ACTIONS))
        action_cat = np.ndarray.tolist(action_cat)
        action_cat.append(fitness_prime)

        return np.asarray(prev_action_cat), np.asarray(action_cat)

    """
    Gets the number of counts after an offspring step and stores them in the haplotype. If a population is reduced to zero then delete it.
    """
    def offspring_step(self):
        haplotypes = list(self.pop.keys())
        counts = self.get_offspring_counts()
        for (haplotype, count) in zip(haplotypes, counts):
            if (count > 0):
                self.pop[haplotype] = count
            else:
                del self.pop[haplotype]

    """
    Returns the new population count for each haplotype given offspring counts weighted by fitness of haplotype
    """
    def get_offspring_counts(self):
        haplotypes = list(self.pop.keys())
        frequencies = [self.pop[haplotype]/self.pop_size for haplotype in haplotypes]
        fitnesses = [self.drug[haplotype] for haplotype in haplotypes]
        weights = [x * y for x,y in zip(frequencies, fitnesses)]
        total = sum(weights)
        weights = [x / total for x in weights]
        return list(np.random.multinomial(self.pop_size, weights))
    

    #####################################################################################################################
    """Shannon Diversity Index"""
    def calc_shannon_diversity(self):
        H = 0
        for i in iter(self.pop.keys()):
            allele_proportion = self.pop[i] / self.pop_size
            H += (allele_proportion * math.log(allele_proportion))
        
        return -H




