#reticulate::use_python("C:/Users/dtw43/AppData/Local/Programs/Python/Python39/python.exe", required = TRUE)
#setup the virtual environment: Do this every time because I've probably changed the code
reticulate::source_python("load.py")
library(dplyr)
library(magrittr)
library(tidyr)
library(stringr)
library(forcats)
library(here)
#define a helper function to clean mems as they come out
clean_mem <- function(mem) {
  mem_mat = matrix(nrow = length(mem), ncol = 5)
  colnames(mem_mat) <- c("ep_number", "action_number", "drug", "reward", "average_fitness")
  for(i in 1:length(mem)) {
    #this indexing is nasty but we have a list of a list of a list so....
    #we end up with a vector in the form [episode number, action number, drug, reward]
    mem_i <- unlist(c(mem[[i]][1], mem[[i]][2],
                      mem[[i]][[3]][[2]], mem[[i]][[3]][[3]], 
                      -mem[[i]][[3]][[3]] + 1)) #the last one computes the average fitness for the period captured by the sensor
    
    mem_mat[i,] <- mem_i
  }
  
  mem_df <- as.data.frame(mem_mat)
  return(mem_df)
}

prep_mdp_mira_sweep <- function(num_evals = 100, episodes = 100, num_steps = 20, 
                                normalize = FALSE) {
  
  
  #run the sweep for - split over ncores
  output = mdp_mira_sweep(num_evals = as.integer(num_evals), 
                          episodes = as.integer(episodes), 
                          num_steps = as.integer(num_steps), 
                          normalize_drugs = normalize)
  mems = output[[1]]
  policies = output[[2]]
  
  out <- list()
  for (i in 1:length(mems)) {
    mem_i <- clean_mem(mems[[i]][[1]])
    mem_i$gamma <- mems[[i]][[2]]
    out[[i]] <- mem_i
  }
  df <- bind_rows(out)
  ##aggregate down so we don't destroy the memory 
  df_prob <- df %>%
    group_by(drug, gamma) %>% 
    summarise(prob_selection = n() / ((num_evals-1)*episodes))
  
  df_sum <- df %>% 
    group_by(gamma) %>% 
    summarise(fitness = mean(average_fitness), 
              sd_fitness = sd(average_fitness))
  
  out <- list(df_prob, df_sum)
  save(out, file = here("data", "results", 
                        paste0(ifelse(normalize, "normalized_", "not_normalized_"),
                               "mdp_sweep.Rda")))
  return(out)
  
}

prep_mdp_sweep <- function(N, sigma_range = list(0, 2), 
                           num_drugs_max=20, episodes=10, num_steps=20,
                           normalize_drugs=FALSE, num_evals=10, save_bool = FALSE) {
  
  #convert everything to integers
  sigma_range = lapply(sigma_range, as.integer)
  num_drugs_max = as.integer(num_drugs_max)
  episodes = as.integer(episodes)
  num_steps = as.integer(num_steps)
  num_evals = as.integer(num_evals)
  N = as.integer(N)
  
  #run the sweep 
  output = mdp_sweep(N=N, sigma_range = sigma_range, 
                     num_drugs_max=num_drugs_max, 
                     episodes=episodes, num_steps=num_steps,
                     normalize_drugs=normalize_drugs, num_evals=num_evals)
  
  mems = output[[1]]
  policies = output[[2]]
  mems_random = output[[3]]
  
  out <- list()
  for (i in 1:length(mems)) {
    #dp solution
    mem_i <- clean_mem(mems[[i]][[1]])
    mem_i$sigma <- mems[[i]][[2]]
    mem_i$num_drugs <- mems[[i]][[3]]
    mem_i$replicate <- mems[[i]][[4]]
    mem_i$condition <- "mdp"
    
    #control group
    mem_random <- clean_mem(mems_random[[i]][[1]])
    mem_random$sigma <- mems_random[[i]][[2]]
    mem_random$num_drugs <- mems_random[[i]][[3]]
    mem_random$replicate <- mems_random[[i]][[4]]
    mem_random$condition <- "random"
    
    #bind, save, iterate
    mem_i <- bind_rows(mem_i, mem_random)
    out[[i]] <- mem_i
  }
  df <- bind_rows(out)
  ##aggregate down so we don't destroy the memory 
  
  if(save_bool) {
    save(df, file = here("data", "results", 
                          paste0(ifelse(normalize, "normalized_", "not_normalized_"),
                                 "mdp_sweep_random.Rda")))
  }
  
  return(df)
  
}

agg_N_sweep <- function(N_range, sigma_range = list(0, 2), 
                        num_drugs_max=20, episodes=10, num_steps=20,
                        normalize_drugs=FALSE, num_evals=10, save_bool = TRUE) {
  mem_list = list()
  for(i in N_range){
    df = prep_mdp_sweep(N=i, sigma_range = sigma_range, 
                         num_drugs_max=num_drugs_max, 
                         episodes=episodes, num_steps=num_steps,
                         normalize_drugs=normalize_drugs, 
                         num_evals=num_evals)
    df$N = i
    mem_list[[i]] <- df
  }
  
  df <- bind_rows(mem_list)
  df_prob <- df %>%
    group_by(drug, sigma, num_drugs, N) %>% 
    summarise(prob_selection = n() / ((num_evals-1)*episodes))
  
  df_sum <- df %>% 
    select(-reward) %>%
    group_by(sigma, num_drugs, N, condition, ep_number) %>% 
    summarise(fitness = mean(average_fitness)) %>%
    pivot_wider(names_from = condition, values_from = fitness) %>%
    mutate(benefit = random - mdp)
  
  
  out = list(df_prob, df_sum,df)
  if(save_bool) {
    save(out, file = here("data", "results", 
                         paste0(ifelse(normalize_drugs, "normalized_", "not_normalized_"),
                                "mdp_sweep_random.Rda")))
  }
  return(out)
}



# out <- agg_N_sweep(N_range = c(4,6,8,10), sigma_range = c(0,5), num_evals = 10, episodes = 50, num_steps = 20, normalize_drugs = FALSE)
# out2 <- agg_N_sweep(N_range = c(4,6,8,10), sigma_range = c(0,5), num_evals = 10, episodes = 50, num_steps = 20, normalize_drugs = TRUE)

out <- prep_mdp_mira_sweep(normalize=FALSE)