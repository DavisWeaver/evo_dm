from landscapes import *
import matplotlib as mpl
import cProfile
import pstats
import time
import random
import itertools


# This is taken from Trevor Bedford's github. It is a Wright-Fisher simulation which includes: Mutation, Selection and Genetic Drift

def main():

    ### Mira landcapes: 
    num_drugs = 15
    drugLandscape = np.zeros((num_drugs,16))
    drugLandscape[0,:]   = [1.851, 2.082, 1.948, 2.434, 2.024, 2.198, 2.033, 0.034, 1.57, 2.165, 0.051, 0.083, 2.186, 2.322, 0.088, 2.821]    #AMP
    drugLandscape[1,:]   = [1.778, 1.782, 2.042, 1.752, 1.448, 1.544, 1.184, 0.063, 1.72, 2.008, 1.799, 2.005, 1.557, 2.247, 1.768, 2.047]    #AM
    drugLandscape[2,:]   = [2.258, 1.996, 2.151, 2.648, 2.396, 1.846, 2.23, 0.214, 0.234, 0.172, 2.242, 0.093, 2.15, 0.095, 2.64, 0.516]      #CEC
    drugLandscape[3,:]   = [0.16, 0.085, 1.936, 2.348, 1.653, 0.138, 2.295, 2.269, 0.185, 0.14, 1.969, 0.203, 0.225, 0.092, 0.119, 2.412]     #CTX
    drugLandscape[4,:]   = [0.993, 0.805, 2.069, 2.683, 1.698, 2.01, 2.138, 2.688, 1.106, 1.171, 1.894, 0.681, 1.116, 1.105, 1.103, 2.591]    #ZOX
    drugLandscape[5,:]   = [1.748, 1.7, 2.07, 1.938, 2.94, 2.173, 2.918, 3.272, 0.423, 1.578, 1.911, 2.754, 2.024, 1.678, 1.591, 2.923]       #CXM
    drugLandscape[6,:]   = [1.092, 0.287, 2.554, 3.042, 2.88, 0.656, 2.732, 0.436, 0.83, 0.54, 3.173, 1.153, 1.407, 0.751, 2.74, 3.227]       #CRO
    drugLandscape[7,:]   = [1.435, 1.573, 1.061, 1.457, 1.672, 1.625, 0.073, 0.068, 1.417, 1.351, 1.538, 1.59, 1.377, 1.914, 1.307, 1.728]    #AMC
    drugLandscape[8,:]   = [2.134, 2.656, 2.618, 2.688, 2.042, 2.756, 2.924, 0.251, 0.288, 0.576, 1.604, 1.378, 2.63, 2.677, 2.893, 2.563]    #CAZ
    drugLandscape[9,:]   = [2.125, 1.922, 2.804, 0.588, 3.291, 2.888, 3.082, 3.508, 3.238, 2.966, 2.883, 0.89, 0.546, 3.181, 3.193, 2.543]    #CTT
    drugLandscape[10,:]  = [1.879, 2.533, 0.133, 0.094, 2.456, 2.437, 0.083, 0.094, 2.198, 2.57, 2.308, 2.886, 2.504, 3.002, 2.528, 3.453]    #SAM
    drugLandscape[11,:]  = [1.743, 1.662, 1.763, 1.785, 2.018, 2.05, 2.042, 0.218, 1.553, 0.256, 0.165, 0.221, 0.223, 0.239, 1.811, 0.288]    #CPR
    drugLandscape[12,:]  = [0.595, 0.245, 2.604, 3.043, 1.761, 1.471, 2.91, 3.096, 0.432, 0.388, 2.651, 1.103, 0.638, 0.986, 0.963, 3.268]    #CPD
    drugLandscape[13,:]  = [2.679, 2.906, 2.427, 0.141, 3.038, 3.309, 2.528, 0.143, 2.709, 2.5, 0.172, 0.093, 2.453, 2.739, 0.609, 0.171]     #TZP
    drugLandscape[14,:]  = [2.59, 2.572, 2.393, 2.832, 2.44, 2.808, 2.652, 0.611, 2.067, 2.446, 2.957, 2.633, 2.735, 2.863, 2.796, 3.203]     #FEP

    
    pop_size = 1000         # Size of Population
    seq_length = 4             # This is "N"
    total_generations = 5000        # How long it runs
    gen_per_step = 500
    steps = int(total_generations / gen_per_step)
    mutation_rate = 1e-5     # per gen per individual per site
    repeats = 100              # Number of landscape replicates - what's this?

    alphabet = ['0', '1']
    base_haplotype = ''.join(["0" for i in range(seq_length)])

    genotypes = [''.join(seq) for seq in itertools.product("01", repeat=seq_length)]

    drug_landscape_dicts = []

    pop = {}
    fitness = {}


    pop[base_haplotype] = pop_size


    for drug in range(num_drugs):
        for i in range(len(genotypes)):
            fitness[genotypes[i]] = drugLandscape[drug,i]

        drug_landscape_dicts.append(copy.deepcopy(fitness))
        fitness.clear()

    
    print(drug_landscape_dicts)
    print(drug_landscape_dicts[0])

    
    history = []

    start = time.time()

    for i in range(steps):
        simulate(pop, history, gen_per_step, mutation_rate, pop_size, seq_length, drug_landscape_dicts[random.randint(0, 14)], alphabet)


    end = time.time()

    print(end-start)

    print(pop)
    



    #calculate fitness
    #avgFit = 0
    #for i in pop.keys():
        #avgFit += (pop[i]/pop_size)*fitness[i]


    #simulateSwitch(pop2, history ,generations, mutation_rate, pop_size, seq_length, fitness, fitnessB, alphabet)
    
    plt.figure(num=None, figsize=(14, 14), dpi=80, facecolor='w', edgecolor='k')
    plt.subplot2grid((3,2), (0,0), colspan=2)
    stacked_trajectory_plot(history, total_generations, pop_size)
    plt.subplot2grid((3,2), (1,0), colspan=2)
    snp_trajectory_plot(history, seq_length, total_generations, pop_size)
    plt.subplot2grid((3,2), (2,0))
    diversity_plot(history, pop_size)
    plt.subplot2grid((3,2), (2,1))
    divergence_plot(history, base_haplotype, pop_size)

    plt.show()

    




"""
Code to put into a separate class/function
"""


#################################################################################################################################
"""
Simulate Wright-Fisher evolution for some amount of generations
"""
def simulate(pop, history, generations, mutation_rate, pop_size, seq_length, fitness, alphabet):
 

def simulateSwitch(pop, history ,generations, mutation_rate, pop_size, seq_length, fitnessA, fitnessB, alphabet):
    clone_pop = dict(pop)
    history.append(clone_pop)

    count = 0
    for i in range(generations):
        if (count < 1):
            time_step(pop, mutation_rate, pop_size, seq_length, fitnessA, alphabet)
            clone_pop = dict(pop)
            history.append(clone_pop)
            count += 1
        else:
            time_step(pop, mutation_rate, pop_size, seq_length, fitnessB, alphabet)
            clone_pop = dict(pop)
            history.append(clone_pop)
            count += 1
            if (count > 2):
                count = 0

"""
Execuse one generation of the Wright-Fisher
"""
def time_step(pop, mutation_rate, pop_size, seq_length, fitness, alphabet):
    mutation_step(pop, mutation_rate, pop_size, seq_length, alphabet)
    offspring_step(pop, pop_size, fitness)


#################################################################################################################################
"""
Below is the code responsible for mutation in the Wright-Fisher model, in order of function calls.
"""

"""
First step of mutation -- get a count of the mutations that occur.
"""
def mutation_step(pop, mutation_rate, pop_size, seq_length, alphabet):
    mutation_count = get_mutation_count(mutation_rate, pop_size, seq_length)
    for i in range(mutation_count):
        mutation_event(pop, pop_size, seq_length, alphabet)

"""
Draw mutation count from a poisson distribution with mean equal to average number of expected mutations.
"""
def get_mutation_count(mutation_rate, pop_size, seq_length):
    mean = mutation_rate * pop_size * seq_length
    return np.random.poisson(mean)


"""
Function that find a random haplotype to mutate and adds that new mutant to the population. Reduces mutated population by 1.
"""
def mutation_event(pop, pop_size, seq_length, alphabet):
    haplotype = get_random_haplotype(pop, pop_size)
    if pop[haplotype] > 1:
        pop[haplotype] -= 1
        new_haplotype = get_mutant(haplotype, seq_length, alphabet)
        if new_haplotype in pop:
            pop[new_haplotype] += 1
        else:
            pop[new_haplotype] = 1

"""
Chooses a random haplotype in the population that will be returned.
"""
def get_random_haplotype(pop, pop_size):
    haplotypes = list(pop.keys())
    frequencies = [x/pop_size for x in pop.values()]
    total = sum(frequencies)
    frequencies = [x / total for x in frequencies]
    return fast_choice(haplotypes, frequencies)
    #return random.choices(haplotypes, weights=frequencies)[0]

"""
Receives the haplotype to be mutated and returns a new haplotype with a mutation with all neighbor mutations equally probable.
"""
def get_mutant(haplotype, seq_length, alphabet):
    site = int(random.random()*seq_length)
    possible_mutations = list(alphabet)
    possible_mutations.remove(haplotype[site])
    mutation = random.choice(possible_mutations)
    new_haplotype = haplotype[:site] + mutation + haplotype[site+1:]
    return new_haplotype


#################################################################################################################################
"""
Below is the code responsible for offspring in the Wright-Fisher model, in order of function calls.
"""


"""
Gets the number of counts after an offspring step and stores them in the haplotype. If a population is reduced to zero then delete it.
"""
def offspring_step(pop, pop_size, fitness):
    haplotypes = list(pop.keys())
    counts = get_offspring_counts(pop, pop_size, fitness)
    for (haplotype, count) in zip(haplotypes, counts):
        if (count > 0):
            pop[haplotype] = count
        else:
            del pop[haplotype]

"""
Returns the new population count for each haplotype given offspring counts weighted by fitness of haplotype
"""
def get_offspring_counts(pop, pop_size, fitness):
    haplotypes = list(pop.keys())
    frequencies = [pop[haplotype]/pop_size for haplotype in haplotypes]
    fitnesses = [fitness[haplotype] for haplotype in haplotypes]
    weights = [x * y for x,y in zip(frequencies, fitnesses)]
    total = sum(weights)
    weights = [x / total for x in weights]
    return list(np.random.multinomial(pop_size, weights))
