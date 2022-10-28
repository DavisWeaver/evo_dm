# #setup the virtual environment: Do this every time because I've probably changed the code
# virtualenv_dir = Sys.getenv('VIRTUALENV_NAME')
# python_path = Sys.getenv('PYTHON_PATH')
# PYTHON_DEPENDENCIES <- "git+https://github.com/DavisWeaver/evo_dm.git"
# # if(reticulate::virtualenv_exists(envname = virtualenv_dir)) {
# #   reticulate::virtualenv_remove(envname = virtualenv_dir, confirm = FALSE)
# # }
# reticulate::virtualenv_create(envname = virtualenv_dir, 
#                               module = "venv")
# reticulate::virtualenv_install(virtualenv_dir, packages = PYTHON_DEPENDENCIES, 
#                                ignore_installed=TRUE)
# reticulate::use_virtualenv(virtualenv_dir, required = T)
#File contains functions to train a load of learners and aggregate the results
library(here)
library(dplyr)
library(reticulate)
library(stringr)
library(magrittr)
library(foreach)
library(tidyr)
source(here("R", "clean_evol_out.R"))
source_python(here("load.py"))

discount_test <- function(episodes = 500, train_input = "fitness", iter = 5){
  
  #load the specified drug list
  N = 4
  n = 1
  num_drugs = 15
  
  #lr_range = 10^(seq(1,6, by = 1))/1e8
  gamma_range = seq(0.001, 0.9999, by  = 0.09)
  
  for(i in 1:length(gamma_range)) {
    for(j in 1:iter) {
      #run evodm
      out <- evol_deepmind(num_evols = 1, N = N, episodes = episodes, 
                           reset_every = 20, min_epsilon = 0.005, 
                           train_input = train_input, 
                           random_start = FALSE, 
                           noise = TRUE, noise_modifier = 1, 
                           num_drugs = num_drugs, 
                           sigma = 0.5, 
                           win_reward = 0, win_threshold = 50, 
                           mira = TRUE, learning_rate = 0.001, 
                           update_target_every = 310, 
                           gamma = gamma_range[i]) #set it up to have no win conditions
      
      cleaned_out <- clean_out(out)
      save(
        cleaned_out, 
        file = here("data", "results", 
                    paste0("mira", i, "N", N, "D", num_drugs, "_gamma", gamma_range[i], "_",
                           ifelse(train_input == "fitness", "fit", "sv"), j, ".Rda"
                    ))
      )
      rm(out)
      gc()
    }
  }
  
  
  return()
  
}

#run for the N5 CS landscapes
# landscapes_test(filename = "CS_landscapes_N5D5.Rda", episodes = 1000, 
#                 train_input = "fitness")
# landscapes_test(filename = "CS_landscapes_N5D5.Rda", episodes = 1000, 
#                 train_input = "state_vector")

#run for N5 random landscapes
# landscapes_test(filename = "random_landscapes_N5D5.Rda", episodes = 4,
#                 train_input = "fitness")
# landscapes_test(filename = "random_landscapes_N5D5.Rda", episodes = 4,
#                 train_input = "state_vector")

#run for mira landscapes
discount_test(episodes = 500,
                train_input = "fitness", iter = 5)
#landscapes_test(mira = TRUE, episodes = 1000,
#                train_input = "state_vector")
