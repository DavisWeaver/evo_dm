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

batch_size_test <- function(episodes = 500, train_input = "fitness", iter = 3){
  
  #load the specified drug list
  N = 4
  n = 1
  num_drugs = 15
  
  #lr_range = 10^(seq(1,6, by = 1))/1e8
  batch_size_range = seq(10,510, by  = 50)
  
  for(i in 1:length(batch_size_range)) {
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
                           minibatch_size = batch_size_range[i]) #set it up to have no win conditions
      
      cleaned_out <- clean_out(out)
      agent <- out[[3]]
      out2 <- evol_deepmind(agent = agent, pre_trained= TRUE)
      agent <- out2[[2]]
      df <- clean_memory(agent)
      mems <- eval_policy_time(agent = agent)
      #put it all together
      out_list <- list(cleaned_out, df, mems)
      
      save(
        out_list, 
        file = here("data", "results", 
                    paste0("mira", i, "N", N, "D", num_drugs, "_batchsize", batch_size_range[i], "_",
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
batch_size_test(episodes = 500,
                train_input = "fitness", iter = 3)
#landscapes_test(mira = TRUE, episodes = 1000,
#                train_input = "state_vector")
