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

landscapes_test <- function(filepath, episodes = 500, train_input = "fitness", 
                            mira = FALSE, iter = 3, reset_every = 20) {
  
  #load the specified drug list
  if(mira) {
    N = 4
    n = 1
    num_drugs = 15
  } else {
    load(filepath)
    N = as.numeric(stringr::str_extract(filepath, "(?<=N)\\d+")) #grab what N should be 
    d = as.numeric(stringr::str_extract(filepath, "(?<=D)\\d+"))
    s = as.numeric(stringr::str_extract(filepath, "(?<=S)\\d+")) #grab what the num drugs should be
    n = length(drug_list)
  }
  
  
  for(i in 1:n) {
    for(j in 1:iter) {
      if(mira) {
        
        #run evodm
        out <- evol_deepmind(num_evols = 1, N = N, episodes = episodes, 
                             reset_every = reset_every, min_epsilon = 0.005, 
                             train_input = train_input, 
                             random_start = FALSE, 
                             noise = TRUE, noise_modifier = 1, 
                             num_drugs = d, 
                             sigma = 0.5, 
                             win_reward = 0, win_threshold = 50, 
                             mira = mira) #set it up to have no win conditions
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
                             reset_every, "_", 
                             ifelse(train_input == "fitness", "fit", "sv"), j, 
                             ".Rda"
                      ))
        )
      } else {
        out <- evol_deepmind(num_evols = 1, N = N, episodes = episodes, 
                             reset_every = 20, min_epsilon = 0.005, 
                             train_input = train_input, 
                             random_start = FALSE, 
                             noise = TRUE, noise_modifier = 1, 
                             num_drugs = d, 
                             sigma = 0.5, 
                             win_reward = 0, win_threshold = 50, 
                             drugs = drug_list[[i]]) #set it up to have no win conditions
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
                      paste0("random", "_landscape", i, "N", N, 
                             "D", d, "S", s, "_", "reset", 
                             reset_every, "_", 
                             ifelse(train_input == "fitness", "fit", "sv"), 
                             j,".Rda"
                      ))
        )
      }
    }
  }
  
  
  return()
  
}

#load all the pre-made landscape filepaths
files <- list.files(here("data", "landscapes"), pattern = "random")
for(i in 1:length(files)) {
  #3 replicates per pre-made landscape. evodm perf
  landscapes_test(filepath = here("data", "landscapes", files[i]), iter = 3, reset_every = 20, 
                  episodes = 500) 
}
