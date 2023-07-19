######Utils####
define_drugs_r <- function(N = 5, num_drugs = 4, 
                           sigma = 0.5, CS = FALSE) {
  landscapes <- generate_landscapes(N = as.integer(N), sigma = sigma)
  drugs <- define_drugs(landscapes, num_drugs = as.integer(num_drugs), CS = CS)
  drugs <- as.list(drugs) #does this make it an R object?
  return(drugs)
}

isLinux <- function() unname(Sys.info()["sysname"]) == "Linux"
######## main function to prep everything to be graphed#######

prep_evol <- function(out) {
  
  df = out[[1]]
  df_naive = out[[2]]
  
  df <- clean_evol(df) %>% 
    mutate(condition = "learner")
  df_naive <- clean_evol(df_naive) %>% 
    mutate(condition = "naive")
  
  df <- bind_rows(df, df_naive)
  
  return(df)
}


#function to clean up reward vector from the evol_deepmind
clean_evol <- function(df) {
  
  #need to go through and name the list elements
  for(i in 1:length(df)) {
    names(df[[i]]) <- c("episode", "average_reward", "min_reward", "max_reward")
  }
  
  df <- bind_rows(df)
  return(df)
}

# Function to prep the fitness landscapes

prep_landscapes <- function(agent) {
  landscapes = agent$env$drugs
}

#function to evaluate the implied policy learned by a replicate after each episode
eval_policy_time <- function(agent){
  
  mems = sweep_replicate_policy(agent= agent)
  
  train_input = agent$env$TRAIN_INPUT
  
  
  mem_list <- list()
  for(i in 1:length(mems)) {
    mem_df <- clean_mem(mems[[i]], train_input = train_input)
    mem_df$original_ep <- i
    mem_list[[i]] <- mem_df
  }
  
  mems <- bind_rows(mem_list)
  
  mems_short = mems %>% 
    group_by(original_ep) %>%
    summarise(fitness= mean(average_fitness))
  mems = list(mems, mems_short)
  return(mems)
  
}



clean_mem <- function(mem, train_input) {
  #Define the internal function of the for loop for this just to keep it tight
  compute_mem_i <- function(mem, train_input) {
    if(train_input == "fitness") {
      mem_i <- unlist(c(mem[[i]][1], mem[[i]][2],
                        mem[[i]][[3]][[2]], mem[[i]][[5]], 
                        -mem[[i]][[5]] + 1, which.max(mem[[i]][[4]]))) #the last one computes the average fitness for the period captured by the sensor
      
    } else {
      mem_i <- unlist(c(mem[[i]][1], mem[[i]][2],
                        mem[[i]][[3]][[2]], mem[[i]][[4]], 
                        -mem[[i]][[4]] + 1, which.max(mem[[i]][[3]][[4]]))) #the last one computes the average fitness for the period captured by the sensor
    }
    return(mem_i)
  }
  mem_mat = matrix(nrow = length(mem), ncol = 6)
  colnames(mem_mat) <- c("ep_number", "action_number", "drug", "average_fitness", "reward", "state")
  for(i in 1:length(mem)) {
    #this indexing is nasty but we have a list of a list of a list so....
    #we end up with a vector in the form [episode number, action number, drug, reward]
    mem_i <- compute_mem_i(mem=mem, train_input=train_input)
    mem_mat[i,] <- mem_i
  }
  
  mem_df <- as.data.frame(mem_mat)
  
  
  ##Now we want to create a master index
  mem_df <- mem_df %>% 
    mutate(index_main = (ep_number - min(ep_number))*max(action_number) + action_number)
  return(mem_df)
}

#function to query and clean up the replay memory
# agent: python class containing information about the agent we trained
# steps: number of time steps to use for computing moving averages, default is 50

clean_memory <- function(agent, steps = 50, pivot = FALSE) {
  
  train_input = agent$env$TRAIN_INPUT
  
  mem = agent$master_memory
  
  #compute mem_df
  mem_df <- clean_mem(mem = mem,train_input = train_input)
  mem_df <- moving_prob(mem_df, steps = steps)
  
  #lets pivot longer
  if(pivot) {
    #extract all columns associated with a given drug
    drug_cols <- colnames(mem_df)[str_detect(colnames(mem_df), "\\d")]
    mem_df <- pivot_longer(mem_df, cols = all_of(drug_cols), 
                           values_to = "moving_prob", 
                           names_to = "drug_regime")
  }
  
  return(mem_df)
}

clean_policies <- function(agent) {
  
  clean_policy_sv <- function(agent) {
    out <- list()
    
    counter = 0
    # a billion for loops to unnest this list monstrosity
    for(i in 1:length(agent$policies)) {
      policy_i <- agent$policies[i][[1]][[1]]
      for(j in 1:length(policy_i)) {
        counter = counter + 1
        actions = unlist(policy_i[[j]])
        df <- data.frame(episode = i, state = j, action = 1:length(actions), prob_selection = actions)
        out[[counter]] <- df
      }
    }
    policies_df <- bind_rows(out)
    
  }
  clean_policy_fit <- function(agent) {
    out <- list()
    
    counter = 0
    # a billion for loops to unnest this list monstrosity
    for(i in 1:length(agent$policies)) {
      policy_i <- agent$policies[i][[1]][[1]]
      
      for(j in 1:length(policy_i)) {
        counter = counter + 1
        
        actions = replicate(n = length(policy_i[[j]]), 0)
        
        action_counts = unlist(policy_i[[j]])
        for(m in 1:length(action_counts)) {
          index <- action_counts[m] + 1 #Plus one to fix the python -R indexing mismatch bs
          actions[index] <- actions[index] +1 #increment 
        }
        
        actions <- actions/length(actions)
        episode <- i
        state <- j
        
        df <- data.frame(episode = i, state = j, action = 1:length(actions), prob_selection = actions)
        out[[counter]] <- df
      }
    }
    
    policies_df <- bind_rows(out)
    
  }
  
  if(agent$env$TRAIN_INPUT == "state_vector") {
    clean_policy = clean_policy_sv
  } else {
    clean_policy = clean_policy_fit
  }
  
  policies_df <- clean_policy(agent)
  
  
  return(policies_df)
}

# function to compute a moving probability that a given action was selected.
#
# Function also computes the moving average of fitness for the same time frame
# Params:
#   df - mem_df
#   steps: size of moving window for computing

moving_prob <- function(df, steps = 50) {
  #gotta sort so its 1234
  actions <- sort(unique(df$drug))
  actions_index <- 1:length(actions) + 2
  
  
  #need to do it for every action
  #initialize output matrix
  out_mat <- matrix(nrow = nrow(df), ncol = 2 + length(actions)) #2 + because the first column of out mat will be the index + we are computing moving probability of fitness as well
  colnames(out_mat) <- c("index_main", "fitness_ma", actions)
  
  for(i in 1:nrow(out_mat)) {
    out_mat[i, "index_main"] <- df$index_main[i]
    if(i <= steps) {
      out_mat[i,colnames(out_mat) != "index_main"] <- NA # don't want to NA the index
    } else {
      moving_window <- df$drug[(i-steps):i]
      #now loop through however many actions
      for(j in 1:length(actions)) { 
        out_mat[i,actions_index[j]] <- sum(moving_window == actions[j])/ length(moving_window)
      }
      out_mat[i, "fitness_ma"] <- mean(df$average_fitness[(i-steps):i]) 
    }
    
  }
  
  out_df <- as.data.frame(out_mat)
  df <- left_join(df, out_df)
  
  return(df)
  
}


clean_out <- function(out) {
  #clean everything
  agent = out[[3]]
  agent_naive = out[[4]]
  agent_dp = out[[5]]
  
  #agent_gene = out_gene[[3]]
  # run the main cleaning function for the reward data
  #df <- prep_evol(out)
  #df_gene <- prep_evol(out_gene)
  
  #clean the master memory
  mem_df <- clean_memory(agent, steps = 150, pivot = TRUE)
  
  #clean for the other conditions
  mem_df_naive <- clean_memory(agent_naive, steps = 150, pivot = TRUE)
  if(agent$hp$WF) {
    mem_df_dp <- list()
  } else {
    mem_df_dp <- clean_memory(agent_dp, steps = 150, pivot = TRUE)
    mem_df_dp$condition = "optimal_policy"
  }
  
  mem_df_naive$condition = "naive"
  mem_df$condition = "evodm"
  
  
  df <- bind_rows(mem_df, mem_df_naive, mem_df_dp)
  
  #add policy info
  policy_df <- clean_policies(agent)
  opt_policy <- out[[7]]
  #add info about optimal policies - this is what we want to save for every run
  return(list(df, policy_df, opt_policy))
}


# Function to compute relevant summary statistics for a given evol_deepmind run
# param: df (mem_df from previous cleaning steps)
#load(file = here("data", "test_mem.Rda")) #if we need to test out some of these functions without re-running the job
agg_dm <- function(df) {
  #find when training chilled out - when do we get to 90% of the minimum fitness achieved
  drug_cols <- colnames(df)[str_detect(colnames(df), "\\d")]
  df <- pivot_wider(df, c("ep_number", "action_number", "index_main"), 
                    names_from = "condition", 
                    values_from = c("drug", "reward", "average_fitness", 
                                    "fitness_ma", all_of(drug_cols))) # hello bug my old friend
  #add a variable for the max absolute fitness decrease achieved relative to the naive agent
  agg_df <- df %>% 
    summarise(
      max_fitness_decrease = 
        min(fitness_ma_naive, na.rm = TRUE) - min(fitness_ma_learner, na.rm = TRUE), 
    )
  
  #add a variable for training time - 
  #its weird because we need to flip the way to stay within 10% of minimum fitness achieved based on the 0.9
  df_trained <- df %>%
    filter(fitness_ma_learner <= 0.9*min(fitness_ma_learner, na.rm = TRUE) |
             fitness_ma_learner <= 1.1*min(fitness_ma_learner, na.rm = TRUE))
  agg_df$actions_to_train <- min(df_trained$index_main)
  
  #add variable describing the learned strategy - is it one-drug dominant, 2, 3, etc
  df_max <- df %>% filter(index_main == max(index_main)) %>% 
    select(-c(reward_learner, average_fitness_learner, fitness_ma_learner, 
              drug_learner, ep_number, action_number), -contains("naive")) %>% 
    pivot_longer(-index_main, names_to = "drug", values_to = "prob_selection")
  
  agg_df <- agg_df %>% 
    mutate(
      learned_strategy = 
        case_when(max(df_max$prob_selection) > 0.9 ~ "one_drug", 
                  sum(df_max$prob_selection > 0.4) == 2 ~ "two_drug",
                  TRUE ~ "threeplus_drug")
    )
  
  #add a variable for average naive fitness for benchmarking
  agg_df$average_naive_fitness <- mean(df$average_fitness_naive)
  agg_df$average_learner_fitness <- mean(df$average_fitness_learner)
  
  return(agg_df)
  
}



