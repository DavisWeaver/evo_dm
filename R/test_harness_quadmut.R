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

landscapes_test <- function(filename, episodes = 1000, train_input = "fitness", 
                            mira = FALSE, iter = 1, reset_every = 20, starting_position) {
  
  #load the specified drug list
  if(mira) {
    N = 4
    n = 1
    num_drugs = 15
  } else {
    load(here("data", filename))
    N = as.numeric(stringr::str_extract(filename, "(?<=N)\\d+")) #grab what N should be 
    num_drugs = as.numeriec(stringr::str_extract(filename, "(?<=D)\\d+")) #grab what the num drugs should be
    n = length(drug_list)
  }
  
  
  for(i in 1:n) {
    if(mira) {
      for(j in 1:iter) {
          #run evodm
          out <- evol_deepmind(num_evols = 1, N = N, episodes = episodes, 
                               reset_every = reset_every, min_epsilon = 0.005, 
                               train_input = train_input, 
                               random_start = FALSE, 
                               noise = TRUE, noise_modifier = 1, 
                               num_drugs = num_drugs, 
                               sigma = 0.5, 
                               win_reward = 0, win_threshold = 50, 
                               mira = mira, starting_genotype = starting_position) #set it up to have no win conditions
          cleaned_out <- clean_out(out)
          
          #run agent through again to simulate performance/ policy over n additional episodes
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
                        paste0("mira", i, "N", N, "D", num_drugs, "_", "reset", 
                               reset_every, "_", "quadmut", 
                               ifelse(train_input == "fitness", "fit", "sv"), j, 
                               ".Rda"
                        ))
          )
        }
    } else {
      out <- evol_deepmind(num_evols = 1, N = N, episodes = episodes, 
                           reset_every = 20, min_epsilon = 0.005, 
                           train_input = train_input, 
                           random_start = FALSE, 
                           noise = TRUE, noise_modifier = 1, 
                           num_drugs = num_drugs, 
                           sigma = 0.5, 
                           win_reward = 0, win_threshold = 50, 
                           drugs = drug_list[[i]], starting_genotype = starting_position) #set it up to have no win conditions
      cleaned_out <- clean_out(out)
      
      agent <- out[[3]]
      out2 <- evol_deepmind(agent = agent, pre_trained= TRUE)
      agent <- out2[[2]]
      df <- clean_memory(agent)
      mems <- eval_policy_time(agent = agent)
      #put it all together
      out_list <- list(cleaned_out, df, mems)
      #save everything
      save(
        out_list, 
        file = here("data", "results", 
                    paste0(
                      ifelse(str_detect(filename, "CS"), "CS", "random"),
                      "_landscape", i, "N", N, "D", num_drugs, "_", "reset", 
                      reset_every, "_", "quadmut", 
                      ifelse(train_input == "fitness", "fit", "sv"), ".Rda"
                    ))
      )
    }
  }
  
  
  return()
  
}

#run for the N5 CS landscagmapes
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
landscapes_test(mira= TRUE, episodes = 500,
                train_input = "fitness", iter = 100, reset_every = 20, 
                starting_position = 15)
#landscapes_test(mira = TRUE, episodes = 1000,
#                train_input = "state_vector")
