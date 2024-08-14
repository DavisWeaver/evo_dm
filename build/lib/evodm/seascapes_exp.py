#import fears
#from fears.experiment import Experiment
#from fears.population import Population
from evodm.evol_game import *
from evodm.data import *

drugs = define_dag_seascapes(file = '../../../evodm_cancer/data/combined_seascapes_cleaned.csv')

env =evol_env(drugs= drugs, seascapes = True, N=4, num_drugs=3, num_conc=10)