from evodm.landscapes import Landscape
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
        #next vgt
        ls.num_jumps = 1
        num_maxes_vgt = len(ls.find_max_indices_alt())
        row_i = {'n':i, 'num_jumps':num_jumps, 'sigma': sigma, 'N':N, 
                    'maxes_hgt': num_maxes_hgt, 'maxes_vgt': num_maxes_vgt}
        
        data.append(row_i)

    df = pd.DataFrame(data)
    return df