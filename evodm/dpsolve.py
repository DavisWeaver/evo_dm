from evodm.evol_game import *
from evodm.landscapes import Landscape
from mdptoolbox.mdp import FiniteHorizon, ValueIteration, PolicyIteration
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
                 num_drugs = 4, drugs = None, noinit = False, phenom = 0,
                 compute_P = False):
        #define the drugs if there aren't any
        if drugs is None:
            ## Generate landscapes - use whatever parameters were set in main()
            landscapes, drugs = generate_landscapes(N = N, sigma = sigma,
                                                    correl = correl, dense = False, 
                                                    num_drugs=num_drugs)
            
            ## Normalize landscapes
            drugs = normalize_landscapes(drugs)
        
        self.drugs = drugs #need these later to compute reward
        #define the landscapes
        landscapes = [Landscape(ls = i, N=N, sigma = sigma, dense = False) for i in drugs]
        #get the transition matrix for each landscape
        #if phenom > 0:
         #   self.tm = [i.get_TM_phenom(phenom) for i in landscapes]
        #else: 
         #   self.tm = [i.get_TM() for i in landscapes]
        self.tm = [i.get_TM_phenom(phenom) for i in landscapes]
        #get number of states
        self.nS = pow(2,N)
        self.nA = num_drugs

        #define initial state distribution
        self.isd = np.zeros(self.nS)
        self.isd[0] = 1

        if noinit:
            return
        
        #define P
        if compute_P:
            self.P = self.define_P()
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
    
    P = env.tm
    R=env.R

    vi= ValueIteration(transitions = P, reward = R, discount = discount_factor, 
                       epsilon = theta)
    vi.run()
    return vi.policy, vi.V

def policy_iteration(env, discount_factor=0.99, theta=0.0001):
    """
    Solve an MDP by policy iteration.
    wrapper of PolicyIteration from mdptoolbox
    """
    P=env.tm
    R=env.R

    pi = PolicyIteration(transitions = P, reward = R, discount = discount_factor)
    pi.run()
    return pi.policy, pi.V
    
