# this file will define the learner class, along with required methods -
# we are taking inspiration (and in some cases borrowing heavily) from the following
# tutorial: https://pythonprogramming.net/training-deep-q-learning-dqn-reinforcement-learning-python-tutorial/?completed=/deep-q-learning-dqn-reinforcement-learning-python-tutorial/

from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv1D, MaxPooling1D, Flatten
from tensorflow.keras.optimizers import Adam
from collections import deque
from .evol_game import evol_env
import random
import numpy as np 
from tqdm import tqdm
from copy import deepcopy

# Function to set hyperparameters for the learner - just edit this any time you
# want to screw around with them .


class hyperparameters:

    def __init__(self):
        # Model training settings
        self.REPLAY_MEMORY_SIZE = 5000
        self.MIN_REPLAY_MEMORY_SIZE = 500
        self.MINIBATCH_SIZE = 100  # still working out what this means
        self.UPDATE_TARGET_EVERY = 500 #every 500 steps, update the target
        self.TRAIN_INPUT = "state_vector"
        
        # Exploration settings
        self.DISCOUNT = 0.9  # also still working out what this really means
        self.epsilon = 1  # lowercase because its not a constant
        self.EPSILON_DECAY = 0.95
        self.MIN_EPSILON = 0.001
        self.LEARNING_RATE = 0.001

        # settings control the evolutionary simulation
        self.NUM_EVOLS = 1 # how many evolutionary steps per time step
        self.SIGMA = 0.5
        self.NORMALIZE_DRUGS = True # should fitness values for all landscapes be bound between 0 and 1?
        # new evolutionary "game" every n steps or n *num_evols total evolutionary movements
        self.RESET_EVERY = 200
        self.EPISODES = 50
        self.N = 5
        self.RANDOM_START = False
        self.NOISE = False #should the sensor readings be noisy?
        self.NOISE_MODIFIER = 1  #enable us to increase or decrease the amount of noise in the system
        self.NUM_DRUGS = 4

        #define victory conditions for player and pop
        self.PLAYER_WCUTOFF = 0.7
        self.POP_WCUTOFF = 0.9

        #define victory threshold
        self.WIN_THRESHOLD = 10 # number of player actions before the game is called
        self.WIN_REWARD = 10

        # stats settings - we aren't really using these.... probably time to remove
        self.AGGREGATE_STATS_EVERY = 1  # AGG EVERY 50 EPISODES
        self.SHOW_PREVIEW = False 


# This is the class for the learning agent
class DrugSelector:

    def __init__(self, hp, drugs = "none"):
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
                            drugs = drugs)

        # main model  # gets trained every step
        self.model = self.create_model()

        # Target model this is what we .predict against every step
        self.target_model = self.create_model()
        self.target_model.set_weights(self.model.get_weights())

        self.replay_memory = deque(maxlen=self.hp.REPLAY_MEMORY_SIZE)
        self.master_memory = []
        self.target_update_counter = 0

    def create_model(self):

        model = Sequential()
        #need to change padding settings if using fitness to train model
        #because sequence may not be long enough
        if self.hp.TRAIN_INPUT == "state_vector":
            model.add(Conv1D(64, 3, activation="relu",
                         input_shape=self.env.ENVIRONMENT_SHAPE))
            model.add(Conv1D(64, 3, activation="relu"))
            model.add(MaxPooling1D(pool_size=2))
        elif self.hp.TRAIN_INPUT == "fitness":
            #have to change the kernel size because of the weird difference in environment shape
            model.add(Conv1D(64, 3, activation="relu",
                         input_shape=self.env.ENVIRONMENT_SHAPE,
                         padding= "same"))
            model.add(Conv1D(64, 3, activation="relu", padding="same"))
            model.add(MaxPooling1D(pool_size=2, padding = "same"))
        elif self.hp.TRAIN_INPUT == "pop_size":
            model.add(Conv1D(64, 3, activation="relu",
                         input_shape=self.env.ENVIRONMENT_SHAPE))
            model.add(Conv1D(64, 3, activation="relu"))
            model.add(MaxPooling1D(pool_size=2))
        else:
            print("please specify either state_vector, fitness, or pop_size for train_input when initializing the environment")
            return
        model.add(Dropout(0.2))
        model.add(Flatten())
        model.add(Dense(128))
        model.add(Dense(len(self.env.ACTIONS), activation="linear"))
        model.compile(loss="mse", optimizer=Adam(learning_rate=self.hp.LEARNING_RATE), metrics=['accuracy'])
        return model

    def update_replay_memory(self):

        #update master memory - for diagnostic purposes only
        self.master_memory.append([self.env.episode_number, self.env.action_number, self.env.sensor])

        if self.hp.NOISE and self.hp.TRAIN_INPUT in ["fitness", "pop_size"]:
            self.add_noise() #adds some stochastic variation to the recorded fitness values
        #update replay memory - this is what the deep learning model has access to
        self.replay_memory.append(self.env.sensor)
    
    def add_noise(self):
        noise_param = np.random.normal(loc = 1.0, 
                                       scale = (0.05 * self.hp.NOISE_MODIFIER))

        #muddy the sensor with the noise param
        self.env.sensor[0] = self.env.sensor[0] * noise_param
        self.env.sensor[3] = self.env.sensor[3] * noise_param
        return 

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
        if self.env.action_number > self.hp.UPDATE_TARGET_EVERY:
            self.target_model.set_weights(self.model.get_weights())
            self.env.action_number = 0

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
        X = np.array(X).reshape(self.hp.MINIBATCH_SIZE, self.env.ENVIRONMENT_SHAPE[0], 1)
        y = np.array(y)

        return X,y
        
    def get_current_states(self, minibatch):
        # Get current states from minibatch, then query NN model for Q values
        current_states = np.array([transition[0] for transition in minibatch])

        #reshape to match expected input dimensions
        current_states = current_states.reshape(self.hp.MINIBATCH_SIZE, 
                                                self.env.ENVIRONMENT_SHAPE[0], 1)

        # Get future states from minibatch, then query NN model for Q values
        # When using target network, query it, otherwise main network should be queried
        new_current_states = np.array(
            [transition[3] for transition in minibatch])
        #reshape to match expected input dimensions
        new_current_states = new_current_states.reshape(self.hp.MINIBATCH_SIZE, 
                                                        self.env.ENVIRONMENT_SHAPE[0], 1)

        return current_states, new_current_states
    #function to get q vector for a given state
    def get_qs(self):
        if self.hp.TRAIN_INPUT == "state_vector":
            tens = self.env.state_vector.reshape(-1, *self.env.ENVIRONMENT_SHAPE)
        elif self.hp.TRAIN_INPUT == "fitness":
            tens = self.env.fitness.reshape(-1, *self.env.ENVIRONMENT_SHAPE)
        elif self.hp.TRAIN_INPUT == "pop_size":
            tens = self.env.pop_size.reshape(-1, *self.env.ENVIRONMENT_SHAPE)
        else: 
            return "error in get_qs()"

        return self.model.predict(tens)[0]

#'main' function that iterates through simulations to train the agent
# practice takes one argument (for now) - an agent of class `DrugSelector`
def practice(agent, naive = False, standard_practice = False):
    #every given number of episodes we are going to track the stats
    #format is [average_reward, min_reward, max_reward]
    reward_list = []
    for episode in tqdm(range(1, agent.hp.EPISODES + 1), ascii=True, unit='episodes'):

        #initialize list of per episode rewards
        ep_rewards = []
        # Restarting episode - reset episode reward and step number
        episode_reward = 0

        for i in range(1, agent.hp.RESET_EVERY):

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
                else:
                    agent.env.action = np.argmax(agent.get_qs()) + 1 #plus one because of the stupid fucking indexing system
            else:
                # Get random action
                if standard_practice:
                        #Only change the action if fitness is above 0.9
                    if np.mean(agent.env.fitness) > 0.9:
                        avail_actions = [action for action in agent.env.ACTIONS if action != agent.env.action] #grab all actions except the one currently selected
                        agent.env.action = random.sample(avail_actions, k = 1)[0] #need to take the first element of the list because thats how random.sample outputs it
                else: 
                    agent.env.action = random.randint(np.min(agent.env.ACTIONS),np.max(agent.env.ACTIONS))


            #we don't save anything - it stays in the class
            agent.env.step()

            # Transform new continous state to new discrete state and count reward
            # can't do this after just 1 step because there won't be anything in the sensor
            if i != 1:
                reward = agent.env.sensor[2]
                episode_reward += reward

                # Every step we update replay memory and train main network - only train if we are doing a not naive run
                agent.update_replay_memory()
                if not naive:
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

        # reset environment for next iteratio
        agent.env.reset()
    return reward_list, agent


def evol_deepmind(num_evols = 1, N = 5, episodes = 50,
                  reset_every = 200, min_epsilon = 0.005,
                  train_input = "fitness",  random_start = False, 
                  noise = False, noise_modifier = 1, num_drugs = 4, 
                  sigma = 0.5, normalize_drugs = True, 
                  player_wcutoff = 0.7, pop_wcutoff = 0.9, win_threshold = 10,
                  win_reward = 10, standard_practice = False, drugs = "none"):
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

    #gotta modulate epsilon decay based on the number of episodes defined
    #0.005 = epsilon_decay^episodes
    hp.EPSILON_DECAY = pow(hp.MIN_EPSILON, 1/hp.EPISODES)


    #initialize agent, including the updated hyperparameters
    agent = DrugSelector(hp = hp, drugs = drugs)
    naive_agent = deepcopy(agent) #otherwise it all gets overwritten by the actual agent

    #run the agent in the naive case and then in the reg case
    naive_rewards, naive_agent = practice(naive_agent, naive = True, standard_practice=standard_practice)
    rewards, agent = practice(agent, naive = False)


    return [rewards, naive_rewards, agent, naive_agent]

#rewards = evol_deepmind()
#naive_rewards= evol_deepmind(naive = True)
