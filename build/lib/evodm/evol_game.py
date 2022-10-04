from evodm.landscapes import Landscape
import numpy as np
import pandas as pd
from tensorflow.keras.utils import to_categorical
import math
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
    def __init__(self, N = 5, sigma = 0.5, correl = np.linspace(-1.0,1.0,51),
                        phenom = 0,
                        train_input = "state_vector", num_evols = 1,
                        random_start = False, 
                        num_drugs = 4, 
                        normalize_drugs = True, 
                        win_threshold = 10,
                        player_wcutoff = 0.8, 
                        pop_wcutoff = 0.95, 
                        win_reward = 10, 
                        drugs = "none", 
                        noise_modifier = 1, 
                        add_noise = True, 
                        average_outcomes = False, 
                        total_resistance = False,
                        starting_genotype = 0):
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


        #data structure for containing information the agent queries from the environment.
        # the format is: [previous_fitness, current_action, reward, next_fitness]
        self.sensor =  []
        self.N = N
        self.sigma = sigma
       
        #Defining these in case self.reset is called
        self.correl = correl
        self.phenom = phenom

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

        #define landscapes
        if drugs == "none": 

            ## Generate landscapes - use whatever parameters were set in main()
            self.landscapes = generate_landscapes(N = N, sigma = sigma,
                                              correl = correl)

            ## Select landscapes corresponding to 4 different drug regimes
            self.drugs = define_drugs(self.landscapes, num_drugs = num_drugs)
        else:
            self.drugs = drugs #use pre-defined drugs 

        #Normalize landscapes if directed
        if normalize_drugs:
            self.drugs = normalize_landscapes(self.drugs)

        ##initialize state vector
        if self.RANDOM_START:
            self.state_vector = np.ones((2**N,1))/2**N
        else:
            self.state_vector  = np.zeros((2**N,1))
            self.state_vector[starting_genotype][0] = 1

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


    def step(self):

        #update current time step of the sim
        #update how many actions have been taken
        self.time_step += self.NUM_EVOLS
        self.action_number += 1
        self.update_target_counter +=1

        # Run the sim under the assigned conditions
        if self.action not in self.ACTIONS:
            return("the action must be in env.ACTIONS")
        fitness, state_vector = run_sim(evol_steps = self.NUM_EVOLS, N = self.N,
                                        sigma = self.sigma,
                                        state_vector = self.state_vector,
                                        drugs = self.drugs, 
                                        action = self.action, 
                                        average_outcomes=self.AVERAGE_OUTCOMES)
        if self.NOISE_BOOL:
            sensor_fitness = self.add_noise(fitness)
        else:
            sensor_fitness = fitness

        
        # Again, this is creating a stacked data structure where each time point provides
        # [current_fitness, current_action, reward, next_fitness]
        if self.action_number != 1:
            if self.TRAIN_INPUT == "state_vector":
                self.sensor = [self.state_vector, self.action, self.calc_reward(fitness = fitness), state_vector]
            elif self.TRAIN_INPUT == "fitness":
                #convert fitness + action into trainable state vector for n and n+1
                prev_action_cat, action_cat = self.convert_fitness(fitness = sensor_fitness)
                self.sensor= [np.array(prev_action_cat), 
                                self.action, self.calc_reward(fitness = fitness), 
                                np.array(action_cat)] 
            elif self.TRAIN_INPUT == "pop_size":
                ##Here we interpolate between the previous fitness and the next fitness
                pop_size = self.growth_curve(new_fitness = fitness)
                self.sensor= [self.pop_size, self.action, self.calc_reward(fitness = fitness), pop_size]
                 #update pop size vector
                self.pop_size = pop_size    
            else:
                print("please specify either state_vector, fitness, or pop_size for train_input when initializing the environment")
                return
        
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
        if self.RANDOM_START:
            self.state_vector = np.ones((2**self.N,1))/2**self.N
        else:
            self.state_vector  = np.zeros((2**self.N,1))
            self.state_vector[self.STARTING_GENOTYPE][0] = 1

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

        

#helper function for generating the od_dist
def s_solve(y):
        x = -math.log(1/y - 1)
        return x

#additional methods used by prep_environ. too lazy to make them part of the class
def generate_landscapes(N = 5, sigma = 0.5, correl = np.linspace(-1.0,1.0,51)):

    A = Landscape(N, sigma)
    Bs = A.generate_correlated_landscapes(correl)

    return Bs

def define_drugs(landscape_to_keep, num_drugs = 4, CS = False):

    if CS: 
        #this code guarantees that high-level CS will be present 
        split_index = np.array_split(range(len(landscape_to_keep)), num_drugs)
        keep_index = [round(np.median(i)) for i in split_index]
    else:
        keep_index = np.random.randint(0, len(landscape_to_keep)-1, size = num_drugs)
    #grab the 'drug landscapes'
    drugs = [landscape_to_keep[i].ls for i in keep_index]

    return drugs

    #pprint.pprint(drugs)           output all the drugs
    #pprint.pprint(drugs[3,0])      output the fitness of genotype 4 in drug 1.

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


def run_sim(evol_steps, N, sigma, state_vector, drugs, action, 
            average_outcomes = False):
    '''
    Function to progress evolutionary simulation forward n times steps in a given fitness regime defined by action

    Args
        evol_steps: int
            number of steps
        N: int
            number of genotypes for the sims
        sigma: float
            constant defining the degree of epistasis on the landscapes
        state_vector: array
            N**2 length array defining the position of the population in genotype space
        drugs: list of lists
            list of n fitness landscapes representing different drug regimes
        action: int
            which drug was selected
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

        # Creates a Landscape object for convenient manipulation and evolution
        landscape_to_evolve = Landscape(N, sigma, ls=drugs[action-1]) #-1 so that it handles pythons stupid dumb indexing system

        # This is the fitness of the population when the drug is selected to be used.
        if not average_outcomes: 
            state_vector = discretize_state(state_vector)

        reward.append(np.dot(landscape_to_evolve.ls,state_vector))  

        # Performs a single evolution step 
        state_vector = landscape_to_evolve.evolve(1, p0=state_vector)
        
    if not average_outcomes:
        state_vector = discretize_state(state_vector) #discretize again before sending it back
        
    reward = np.squeeze(reward)
    return reward, state_vector


#Function to compute reward for a given simulation step - used by the environment class. 
#Could have defined this in-line but made it a separate function in case we want to make it 
#more sophisticated in the future. 

