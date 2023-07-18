from evodm.landscapes import Landscape
from evodm.evol_game import evol_env_wf, generate_landscapes, normalize_landscapes
from evodm.data import get_example_drug
import pandas as pd
import numpy as np

def num_peaks(N, sigma, n, num_jumps):
    """
    num_peaks counts the number of peaks in n arbitrary landscapes.

    ...

    Args
    ------
    N: int 
        number of adaptations
    sigma: float
        epistasis coefficient
    n: int
        number of landscapes
    num_jumps: int
        number of adaptations that can be transferred or lost in a single time step in the hgt condition
    """

    data = []
    for i in range(n):
        #first hgt
        ls = Landscape(N=N, sigma=sigma, num_jumps=num_jumps)
        num_maxes_hgt = len(ls.find_max_indices_alt())
        num_edges_hgt = ls.get_total_edges()
        #next vgt
        ls.num_jumps = 1
        num_maxes_vgt = len(ls.find_max_indices_alt())
        num_edges_vgt = ls.get_total_edges()
        row_i = {'n':i, 'num_jumps':num_jumps, 'sigma': sigma, 'N':N, 
                    'maxes_hgt': num_maxes_hgt, 'maxes_vgt': num_maxes_vgt, 
                    'edges_hgt': num_edges_hgt, 'edges_vgt': num_edges_vgt}
        
        data.append(row_i)

    df = pd.DataFrame(data)
    return df

def run_sim_hgt(max_evol_steps, ls):
    
    '''
    Function to progress evolutionary simulation forward until the population reaches a local maxima and stops evolving

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
    state_vector  = np.zeros((2**ls.N,1))
    state_vector[0][0] = 1
    # Evolve for 100 steps.
    for i in range(max_evol_steps):
        old_state_vector = state_vector
        # Performs a single evolution step 
        state_vector = ls.evolve(1, p0=state_vector)
         # This is the fitness of the population when the drug is selected to be used.
        if all(state_vector == old_state_vector):
            break

    fit = np.dot(ls.ls,state_vector) 
    max_fit = np.max(ls.ls)
    prop_global_max = state_vector[ls.find_global_max()]
    return i, fit, max_fit, prop_global_max
        

def summarize_wf_hgt(N=5, sigma = 0.5, num_drugs = 5, pop_size = 10000, 
                     gen_per_step = 20, mutation_rate = 1e-5, hgt_rate= 1e-5, 
                     theta = 1e-5, stop_count = 20):
    ls, drugs = generate_landscapes(N=N, sigma = sigma, num_drugs=num_drugs)
    drugs = normalize_landscapes(drugs) #can't have any non 0 fitnesses or everyone loses their gd mind

    env = evol_env_wf(N =N, num_drugs = num_drugs, pop_size = pop_size, 
                      gen_per_step=gen_per_step,
                      mutation_rate=mutation_rate, hgt_rate = hgt_rate,
                      drugLandscape=drugs)
    
    env.drug = get_example_drug(N=N)
    #a procedurally generated drug landscape where the best allele is very far from the starting point
    #hard coding it here just so all versions of this summary analysis use the same landscape
    #jk I'm going to load it from a different function - should do disk but I'm too lazy
    
    data = []
    max_fitness = np.max(drugs[0])
    end_counter = 0
    end_counter = 0
    for i in range(10000):
        if i != 0:
            old_fit = fitness
        env.time_step()
        fitness = env.compute_pop_fitness(drug = env.drug, sv = env.pop)
        shannon_index = env.calc_shannon_diversity()
        dominant_pop = max(env.pop, key=env.pop.get)
        num_pops = len(env.pop)
        data.append({'N':N, 'sigma':sigma, 'pop_size':pop_size, 
                     'mutation_rate':mutation_rate, 
                     'hgt_rate': hgt_rate,
                     'step_num':i,
                     'fitness':fitness,
                     'shannon_index':shannon_index, 
                     'dom_allele': dominant_pop, 
                     'num_alleles':num_pops})
        
        #check if we can stop
        if fitness > 0.9999:
            break
        if i != 0:
            #need to set this up so if it stops improving for n time steps
            delta = abs(fitness - old_fit)
            if delta < theta:
                end_counter+=1
                if end_counter >= stop_count: #if fitness is static for 10 time steps in a row, stop the count
                    break
            else:
                end_counter=0

    df = pd.DataFrame(data)
    return [df, env]
    



