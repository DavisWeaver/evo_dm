from evodm.learner import compute_optimal_policy, compute_optimal_action, DrugSelector, practice, hyperparameters
from evodm.dpsolve import backwards_induction, dp_env
from evodm.exp import define_mira_landscapes
import random
import numpy as np
import pytest
from itertools import chain
from copy import deepcopy

hp = hyperparameters()
hp.TRAIN_INPUT = "fitness"
hp.MIN_REPLAY_MEMORY_SIZE = 20
hp.MINIBATCH_SIZE = 10
hp.RESET_EVERY = 20
hp.EPISODES = 10
hp.N = 4
hp.NUM_DRUGS = 15
hp.NOISE=True
hp.NOISE_MODIFIER=5
hp.AVERAGE_OUTCOMES = False

drugs = define_mira_landscapes()
hp.NUM_DRUGS = 15
hp.RESET_EVERY = 20
hp.NORMALIZE_DRUGS = False
ds = DrugSelector(hp = hp, drugs = drugs)

practice(agent = ds, train_freq = 100, compute_implied_policy_bool=False)
