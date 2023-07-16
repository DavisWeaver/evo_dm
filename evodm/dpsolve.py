from evodm.evol_game import *
from evodm.landscapes import Landscape
from mdptoolbox.mdp import FiniteHorizon
import numpy as np

class dp_env:
    """
    Contains the required data to setup MDP and perform policy eval/ policy improvement.
    
    Args:
        drugs: list of lists - fitness landscapes
        N: Number of genotypes
        sigma: epistasis coefficient
        
    Returns:
        a class containing the following required data: 
        dp_env.P: transition tuples where dp_env.P[s][a] corresponds to all transitions for a given state-action pair.
        dp_env.nA: number of actions
        dp_env.nS: number of states
    """
    def __init__(self, N, sigma, 
                 correl = np.linspace(-1.0,1.0,51), 
                 num_drugs = 4, drugs = "none", noinit = False):
        #define the drugs if there aren't any
        if drugs == "none":
            ## Generate landscapes - use whatever parameters were set in main()
            landscapes = generate_landscapes(N = N, sigma = sigma,
                                              correl = correl, dense = False)

            ## Select landscapes corresponding to 4 different drug regimes
            drugs = define_drugs(landscapes, num_drugs = num_drugs)
            ## Normalize landscapes
            drugs = normalize_landscapes(drugs)
        
        self.drugs = drugs #need these later to compute reward
        #define the landscapes
        landscapes = [Landscape(ls = i, N=N, sigma = sigma, dense = False) for i in drugs]
        #get the transition matrix for each landscape
        self.tm = [i.get_TM() for i in landscapes]
        
        #get number of states
        self.nS = pow(2,N)
        self.nA = num_drugs

        #define initial state distribution
        self.isd = np.zeros(self.nS)
        self.isd[0] = 1

        if noinit:
            return
        
        #define P
        #self.P = self.define_P()
        #define R
        self.R = self.define_R()
        self.tm = self.clean_tm()
        
        
    
    def define_P(self):
        """
        Method to define transition tuples for that define the markov decision process
    
        Args:
            self: inherits necessary arguments from parent class
        
        Returns:
            A dict of dicts where P[s][a] = [(prob, next_state, reward, is_done)]
            P[s][a] typically returns a list of tuples corresponding to all possible next states given a state-action pair.
        """
        P = {}
        # P[s][a] = [(prob, next_state, reward, is_done)] for all next_states
        for s in range(self.nS):
            P[s] = {a : [] for a in range(self.nA)}
            for a in range(self.nA):
                p_list = []
                #this gets us a 1d list of transition probabilities from s --> s' for all s' in S
                tps = [i[s] for i in self.tm[a]]
                
                #iterate through all states S to generate transition tuples s --> s' [(prob, next_state, reward, is_done)]
                for s_prime in range(self.nS):
                    tp = tps[s_prime]
                    if tp != 0:
                        #retrieve fitness for a given action, s' pair and compute reward (1-fitness)
                        reward = 1 - self.drugs[a][s_prime] 
                        p_list.append((tp, s_prime, reward, False))
                
                P[s][a] = p_list 
                        
        return P
    
    def define_R(self):
        """
        Method to define a reward matrix R
    
        Args:
            self: inherits necessary arguments from parent class
        
        Returns:
            List of lists where R[a][s] yields r(s|a)
        """
        R = []
        for s in range(self.nS):
            R_s = []
            for j in range(len(self.drugs)):
                R_j = 1 - self.drugs[j][s]
                R_s.append(R_j)
            R_s = np.asarray(R_s)
            R.append(R_s)
        R = np.asarray(R)
        return R

    def clean_tm(self):
        """
        Method to transpose the transition matrix for each action a 
        because Jeff defined it in a dumb way
        Args:
            self: inherits necessary arguments from parent class
        
        Returns:
           array of matrices where tm[A] yields p(s'|s)
        """
        new_tm = []
        for a in range(self.nA):
            tm_a = np.transpose(self.tm[a])
            new_tm.append(tm_a)
        tm = np.asarray(new_tm)
        return tm

        


def backwards_induction(env, num_steps = 20, discount_rate = 0.99):
    """
    Backwards induction for finite horizon MDPs. Mostly a compatibility function for mdptoolbox.mdp.FiniteHorizon
    
    Args:
        env: dp_env defined above - defines the markov decision process 
            for the problem of drug cycling to treat a population evolving on fitness landscapes
            env.P[s][a] is a list of transition tuples (prob, next_state, reward, done).
            env.nS is a number of states in the environment. 
            env.nA is a number of actions in the environment.
        num_steps: what is the time horizon? defaults to 20
        discount_factor: Gamma discount factor.
        
    Returns:
        A tuple (policy, V) of the optimal policy and the optimal value function.
    """
    P = env.tm
    R = env.R

    fh = FiniteHorizon(transitions = P, reward = R, 
                                   discount = discount_rate, N = num_steps)

    fh.run()

    return fh.policy, fh.V


def value_iteration(env, theta=0.0001, discount_factor=0.99):
    """
    Value Iteration Algorithm.
    
    Args:
        env: dp_env defined above - defines the markov decision process 
            for the problem of drug cycling to treat a population evolving on fitness landscapes
            env.P[s][a] is a list of transition tuples (prob, next_state, reward, done).
            env.nS is a number of states in the environment. 
            env.nA is a number of actions in the environment.
        theta: We stop evaluation once our value function change is less than theta for all states.
        discount_factor: Gamma discount factor.
        
    Returns:
        A tuple (policy, V) of the optimal policy and the optimal value function.
    """
    
    def one_step_lookahead(state, V):
        """
        Helper function to calculate the value for all action in a given state.
        
        Args:
            state: The state to consider (int)
            V: The value to use as an estimator, Vector of length env.nS
        
        Returns:
            A vector of length env.nA containing the expected value of each action.
        """
        A = np.zeros(env.nA)
        for a in range(env.nA):
            for prob, next_state, reward, done in env.P[state][a]:
                A[a] += prob * (reward + discount_factor * V[next_state])
        return A
    
    V = np.zeros(env.nS)
    while True:
        # Stopping condition
        delta = 0
        # Update each state...
        for s in range(env.nS):
            # Do a one-step lookahead to find the best action
            A = one_step_lookahead(s, V)
            best_action_value = np.max(A)
            # Calculate delta across all states seen so far
            delta = max(delta, np.abs(best_action_value - V[s]))
            # Update the value function. Ref: Sutton book eq. 4.10. 
            V[s] = best_action_value        
        # Check if we can stop 
        if delta < theta:
            break
    
    # Create a deterministic policy using the optimal value function
    policy = np.zeros([env.nS, env.nA])
    for s in range(env.nS):
        # One step lookahead to find the best action for this state
        A = one_step_lookahead(s, V)
        best_action = np.argmax(A)
        # Always take the best action
        policy[s, best_action] = 1.0
    
    return policy, V

def policy_eval(policy, env, discount_factor=0.99, theta=0.00001):
    """
    Evaluate a policy given an environment and a full description of the environment's dynamics.
    
    Args:
        policy: [S, A] shaped matrix representing the policy.
        env: OpenAI env. env.P represents the transition probabilities of the environment.
            env.P[s][a] is a list of transition tuples (prob, next_state, reward, done).
            env.nS is a number of states in the environment. 
            env.nA is a number of actions in the environment.
        theta: We stop evaluation once our value function change is less than theta for all states.
        discount_factor: Gamma discount factor.
    
    Returns:
        Vector of length env.nS representing the value function.
    """
    # Start with a random (all 0) value function
    V = np.zeros(env.nS)
    while True:
        delta = 0
        # For each state, perform a "full backup"
        for s in range(env.nS):
            v = 0
            # Look at the possible next actions
            for a, action_prob in enumerate(policy[s]):
                # For each action, look at the possible next states...
                for  prob, next_state, reward, done in env.P[s][a]:
                    # Calculate the expected value
                    v += action_prob * prob * (reward + discount_factor * V[next_state])
            # How much our value function changed (across any states)
            delta = max(delta, np.abs(v - V[s]))
            V[s] = v
        # Stop evaluating once our value function change is below a threshold
        if delta < theta:
            break
    return np.array(V)

def policy_improvement(env, policy_eval_fn=policy_eval, discount_factor=0.99):
    """
    Policy Improvement Algorithm. Iteratively evaluates and improves a policy
    until an optimal policy is found.
    
    Args:
        env: dp_env defined above - defines the markov decision process 
            for the problem of drug cycling to treat a population evolving on fitness landscapes
        policy_eval_fn: Policy Evaluation function that takes 3 arguments:
            policy, env, discount_factor.
        discount_factor: gamma discount factor.
        
    Returns:
        A tuple (policy, V). 
        policy is the optimal policy, a matrix of shape [S, A] where each state s
        contains a valid probability distribution over actions.
        V is the value function for the optimal policy.
        
    """

    def one_step_lookahead(state, V):
        """
        Helper function to calculate the value for all action in a given state.
        
        Args:
            state: The state to consider (int)
            V: The value to use as an estimator, Vector of length env.nS
        
        Returns:
            A vector of length env.nA containing the expected value of each action.
        """
        A = np.zeros(env.nA)
        for a in range(env.nA):
            for prob, next_state, reward, done in env.P[state][a]:
                A[a] += prob * (reward + discount_factor * V[next_state])
        return A
    
    # Start with a random policy
    policy = np.ones([env.nS, env.nA]) / env.nA
    
    while True:
        # Evaluate the current policy
        V = policy_eval_fn(policy, env, discount_factor)
        
        # Will be set to false if we make any changes to the policy
        policy_stable = True
        
        # For each state...
        for s in range(env.nS):
            # The best action we would take under the current policy
            chosen_a = np.argmax(policy[s])
            
            # Find the best action by one-step lookahead
            # Ties are resolved arbitarily
            action_values = one_step_lookahead(s, V)
            best_a = np.argmax(action_values)
            
            # Greedily update the policy
            if chosen_a != best_a:
                policy_stable = False
            policy[s] = np.eye(env.nA)[best_a]
        
        # If the policy is stable we've found an optimal policy. Return it
        if policy_stable:
            return policy, V

