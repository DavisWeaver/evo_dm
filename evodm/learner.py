# this file will define the learner class, along with required methods -
# we are taking inspiration (and in some cases borrowing heavily) from the following
# tutorial: https://pythonprogramming.net/training-deep-q-learning-dqn-reinforcement-learning-python-tutorial/?completed=/deep-q-learning-dqn-reinforcement-learning-python-tutorial/
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Conv1D, MaxPooling1D, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.python.keras.layers import deserialize, serialize
from tensorflow.python.keras.saving import saving_utils
from tensorflow.keras.utils import to_categorical
from collections import deque
from evodm.evol_game import evol_env, evol_env_wf
from evodm.dpsolve import backwards_induction, dp_env
import random
import numpy as np 
from copy import deepcopy
from tqdm import tqdm 

# Function to set hyperparameters for the learner - just edit this any time you
# want to screw around with them.
#or edit directly

def unpack(model, training_config, weights):
    restored_model = deserialize(model)
    if training_config is not None:
        restored_model.compile(
            **saving_utils.compile_args_from_training_config(
                training_config
            )
        )
    restored_model.set_weights(weights)
    return restored_model

# Hotfix function
def make_keras_picklable():

    def __reduce__(self):
        model_metadata = saving_utils.model_metadata(self)
        training_config = model_metadata.get("training_config", None)
        model = serialize(self)
        weights = self.get_weights()
        return (unpack, (model, training_config, weights))

    cls = Model
    cls.__reduce__ = __reduce__

# Run the function
make_keras_picklable()

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
        self.LEARNING_RATE = 0.0001

        # settings control the evolutionary simulation
        self.NUM_EVOLS = 1 # how many evolutionary steps per time step
        self.SIGMA = 0.5
        self.NORMALIZE_DRUGS = True # should fitness values for all landscapes be bound between 0 and 1?
        self.AVERAGE_OUTCOMES = False #should we use the average of infinite evolutionary sims or use a single trajectory?
        # new evolutionary "game" every n steps or n *num_evols total evolutionary movements
        self.RESET_EVERY = 20
        self.EPISODES = 500
        self.N = 5
        self.RANDOM_START = False
        self.STARTING_GENOTYPE = 0 #default to starting at the wild type genotype
        self.NOISE = False #should the sensor readings be noisy?
        self.NOISE_MODIFIER = 1  #enable us to increase or decrease the amount of noise in the system
        self.NUM_DRUGS = 4
        self.MIRA = True
        self.TOTAL_RESISTANCE = False
        #wright-fisher controls
        self.WF = False
        self.POP_SIZE = 10000
        self.GEN_PER_STEP = 1
        self.MUTATION_RATE = 1e-5

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
        if self.hp.WF:
            self.env = evol_env_wf(train_input = self.hp.TRAIN_INPUT,
                                   pop_size = self.hp.POP_SIZE, 
                                   gen_per_step = self.hp.GEN_PER_STEP, 
                                   mutation_rate = self.hp.MUTATION_RATE)
        else:
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
                                starting_genotype = self.hp.STARTING_GENOTYPE,
                                total_resistance= self.hp.TOTAL_RESISTANCE)

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
                self.master_memory.append([self.env.episode_number, self.env.action_number, self.env.sensor, self.env.state_vector, self.env.fitness])
            else:
                self.master_memory.append([self.env.episode_number, self.env.action_number, self.env.sensor, self.env.fitness]) #also record real fitness instead of sensor fitness
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

        current_qs_list = self.model.predict(current_states, verbose = 0)
        future_qs_list = self.target_model.predict(new_current_states, verbose = 0)

        
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
                    if self.hp.WF:
                        fit = np.dot(list(self.env.drugs[a].values()), state_vector)[0] #compute fitness for given state_vector, drug combination
                    else:
                        fit = np.dot(self.env.drugs[a], state_vector)[0] #compute fitness for given state_vector, drug combination
                    a_vec = deepcopy(a_list)[a]
                    #append fitness to one-hot encoded action to mimic how the data are fed into the model
                    a_vec.append(fit)
                    a_vec = np.array(a_vec)
                    #reshape to feed into the model
                    tens = a_vec.reshape(-1, *self.env.ENVIRONMENT_SHAPE)
                    #find the optimal action
                    action_a = self.model.predict(tens, verbose = 0)[0].argmax()
            
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

        return self.model.predict(tens, verbose = 0)[0]
    
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

    return policy,V

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
        if agent.env.TRAIN_INPUT == "state_vector": 
            action = np.argmax(policy[index]) + 1 
        else:
            action = policy[index][int(agent.env.prev_action)-1] +1#plus one because I made the bad decision to force the actions to be 1-indexed once upons a time
    else:
        action = policy[index][step] + 1 #plus one because I made the bad decision to force the actions to be 1,2,3,4 once upon a time
    
    return action
    
#'main' function that iterates through simulations to train the agent
def practice(agent, naive = False, standard_practice = False, 
             dp_solution = False, pre_trained = False, discount_rate = 0.99,
             policy = "none", prev_action = False, wf = False, train_freq = 100, 
             compute_implied_policy_bool = False):
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
    train_freq: int
        how many time steps should pass between training the model. 

    Returns rewards, agent, policy 
        reward vector, trained agent including master memory dictating what happened, and learned policy (if applicable)
    '''
    if dp_solution and not wf:
        dp_policy, V = compute_optimal_policy(agent, discount_rate = discount_rate,
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
    count=1
    for episode in tqdm(range(1, agent.hp.EPISODES + 1), ascii=True, unit='episodes', 
                        disable = True if any([dp_solution, naive, pre_trained]) else False):
        # Restarting episode - reset episode reward and step number
        episode_reward = 0
        if pre_trained:
            agent.hp.epsilon = 0

        for i in range(agent.hp.RESET_EVERY+1):
            if i==0:
                agent.env.step()
                continue
            i_fixed =i-1 #correct for the drastic step we had to take up above ^
            # This part stays mostly the same, the change is to query a model for Q values
            if np.random.random() > agent.hp.epsilon:
                # Get action from Q table
                if naive:
                    if standard_practice and not wf:
                        #Only change the action if fitness is above 0.9
                        if np.mean(agent.env.fitness) > 0.9:
                            avail_actions = [action for action in agent.env.ACTIONS if action != agent.env.action] #grab all actions except the one currently selected
                            agent.env.action = random.sample(avail_actions, k = 1)[0] #need to take the first element of the list because thats how random.sample outputs it
                    else: 
                        if wf:
                            agent.env.update_drug(random.randint(np.min(agent.env.ACTIONS),np.max(agent.env.ACTIONS)))
                        else:
                            agent.env.action = random.randint(np.min(agent.env.ACTIONS),np.max(agent.env.ACTIONS))
                elif dp_solution:
                    agent.env.action = compute_optimal_action(agent, dp_policy, step = i_fixed, prev_action=prev_action)
                else:
                    if wf:
                        agent.env.update_drug(np.argmax(agent.get_qs()))
                    else:
                        agent.env.action = np.argmax(agent.get_qs()) + 1 #plus one because of the stupid fucking indexing system
            else:
                # Get random action
                if standard_practice and not wf:
                        #Only change the action if fitness is above 0.9
                    if np.mean(agent.env.fitness) > 0.9:
                        avail_actions = [action for action in agent.env.ACTIONS if action != agent.env.action] #grab all actions except the one currently selected
                        agent.env.action = random.sample(avail_actions, k = 1)[0] #need to take the first element of the list because thats how random.sample outputs it
                elif dp_solution:
                    agent.env.action = compute_optimal_action(agent, dp_policy, step = i_fixed, prev_action = prev_action)
                elif wf:
                    agent.env.update_drug(random.randint(np.min(agent.env.ACTIONS),np.max(agent.env.ACTIONS)))
                else: 
                    agent.env.action = random.randint(np.min(agent.env.ACTIONS),np.max(agent.env.ACTIONS))


            #we don't save anything - it stays in the class
            agent.env.step()

            reward = agent.env.sensor[2]
            episode_reward += reward

            # Every step we update replay memory and train main network - only train if we are doing a not naive run
            agent.update_replay_memory()

            if not any([dp_solution, naive, pre_trained]):
                if count % train_freq == 0: #this will prevent us from training every freaking time step
                    agent.train()
                    if train_freq > agent.hp.RESET_EVERY and compute_implied_policy_bool:
                        if not agent.hp.NUM_EVOLS > 1:
                            agent.compute_implied_policy(update = True)

            if agent.env.done: # break if either of the victory conditions are met
                break #check out calc_reward in the evol_env class for how this is defined

            count +=1 #keep track of total number of time steps that pass

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
                if not agent.hp.NUM_EVOLS > 1 and compute_implied_policy_bool:
                    if not train_freq > agent.hp.RESET_EVERY:
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
        V=[]
    elif pre_trained:
        policy = []
        V=[]
    elif compute_implied_policy_bool:
        policy = agent.compute_implied_policy(update = False)
        V=[]
    else:
        policy = []
        V = []
    return reward_list, agent, policy, V



