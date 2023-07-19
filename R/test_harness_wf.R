#File contains functions to train a load of learners and aggregate the results

reticulate::use_python("C:/Program Files/Python37/python.exe", required = TRUE) #python path may vary
#^^ comment this out if not running locally at the office ^^ 
library(here)
library(dplyr)
library(reticulate)
library(stringr)
library(magrittr)
library(foreach)
library(tidyr)
source(here("R", "clean_evol_out.R"))
source_python(here("load.py"))

landscapes_test <- function(episodes = 500, train_input = "fitness", 
                            iter = 1) {
  
  #load the specified drug list
  
  N = 4
  d = 15
  
  for(j in 1:iter) {
    #run evodm
    out <- evol_deepmind(num_evols = 1, N = N, episodes = episodes, 
                         reset_every = 10000, min_epsilon = 0.005, 
                         train_input = train_input,
                         wf = TRUE,
                         random_start = FALSE, 
                         gen_per_step = as.integer(1),
                         noise = FALSE, 
                         num_drugs = d, mira=  TRUE) #set it up to have no win conditions
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
                  paste0("mira_wf_", "N", N, "D", num_drugs, "_", "reset", 
                         reset_every, "_", 
                         ifelse(train_input == "fitness", "fit", "sv"), j, 
                         ".Rda"
                  ))
    )
  }
  
  return()
  
}

#load all the pre-made landscape filepaths
files <- list.files(here("data", "landscapes"), pattern = "random")
for(i in 1:length(files)) {
  #3 replicates per pre-made landscape. evodm perf
  landscapes_test(iter = 1,
                  episodes = 500) 
}
