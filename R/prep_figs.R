
#Figure that looks at policies for a given agent
prep_performance_overview <- function(N=5, landscapes = "random", files = NULL, 
                                      lr_range= NULL, noise_exp = FALSE, policy_exists = TRUE, dir = "results") {
  
  
  add_filestring_vars <- function(file, df, noise_exp = FALSE) {
    
    df$landscapes <- str_extract(file, "\\d+(?=N)") #grab the landscape set from the filename
    train_input <- str_extract(file, "[:alpha:]{2,3}(?=\\d+\\.Rda)")
    df$train_input <- train_input
    df$replicate <- str_extract(file, paste0("(?<=", train_input,")\\d+"))
    df$totalres <- str_detect(file, "totalres")
    df$quadmut <- str_detect(file, "quadmut")
    
    if(noise_exp) {
      df$noise_modifier <- str_extract(file, "(?<=M)\\d+")
    }
    
    
    if(landscapes == "lr") {
      lr_num <- as.numeric(str_extract(file, "(?<=lr)\\d+"))
      df$lr <- lr_range[lr_num]
      
    }
    
    if(landscapes == "updatetarget") {
      df$update_target <- as.numeric(str_extract(file, "(?<=updatetarget)\\d+"))
    }
    
    if(landscapes == "batchsize") {
      df$batch_size <- as.numeric(str_extract(file, "(?<=batchsize)\\d+"))
    }
    if(landscapes == "gamma") {
      df$gamma <- as.numeric(str_extract(file, "(?<=gamma)\\d+\\.\\d+"))
    }
    
    if(landscapes == "mira") {
      df$reset <- str_extract(files[i], "(?<=reset)\\d+")
      df$replicate_mod <- str_extract(file, "(?<=mira)\\d+")
    }
    return(df)
  }
  
  #check if the user provided a vector of filenames
  if(is.null(files)) {
    files <- list.files(here("data", dir), pattern = landscapes)
  }
  #omit the hyperparameter sweeps if we are looking to prep the mira heatmap figure
  if(landscapes == "mira") {
    files <- files[!grepl("lr\\d", files)] #totalres has an lr in it hilariously
    files <- files[!grepl("updatetarget", files)]
    files <- files[!grepl("batchsize", files)]
    files <- files[!grepl("gamma", files)]
  }
  
  if(noise_exp) {
    files <- files[grepl("M\\d", files)]
  } else {
    files <- files[!grepl("M\\d", files)]
  }
  df_list <- list()
  policies <- list()
  mems <- list()
  for(i in 1:length(files)) {
    #clean  the master_memory dataframes
    load(here("data", dir, files[i]))
    
    test <- try(out_list[[1]], silent = TRUE)
    if(inherits(test, 'try-error')) {
      next
      df = cleaned_out[[1]] %>%
        tidyr::pivot_wider(names_from = drug_regime, values_from = moving_prob)
      df <- add_filestring_vars(file = files[i], df, noise_exp=noise_exp)
    } else {
      cleaned_out <- out_list[[1]]
      df <- out_list[[2]]
      df <- add_filestring_vars(file = files[i], df, noise_exp = noise_exp) 
      df$condition = "evodm"
      
      
      #these are the other conditions
      df2 <- cleaned_out[[1]] 
      df2 <- add_filestring_vars(file = files[i], df2, noise_exp = noise_exp) %>% 
        dplyr::filter(condition != "evodm") %>% 
        tidyr::pivot_wider(names_from = drug_regime, values_from = moving_prob)
      
      df <- bind_rows(df, df2)
    }
    
    #pivot so we don't have num_drugs * the number of obs we want
    #df <- tidyr::pivot_wider(df, names_from = drug_regime, values_from = moving_prob)
    df_list[[i]] <- df
    
    opt_policy <- cleaned_out[[3]] %>% 
      as.data.frame() %>% 
      mutate(state = 1:nrow(cleaned_out[[3]])) %>% 
      pivot_longer(cols = -state, 
                   names_to = "action", 
                   names_prefix = "V", 
                   values_to = "optimal_action") %>%
      mutate(action = as.numeric(action))
    
    if(policy_exists) {
      #join the policy dfs.
      policy <- cleaned_out[[2]]
      policy <- add_filestring_vars(file = files[i], df = policy, noise_exp = noise_exp)
      #now handle the optimal policies
      
      policy <- left_join(policy, opt_policy)
      
      policies[[i]] <- policy
      mems_i <- out_list[[3]]
      mems_i <- add_filestring_vars(file = files[i], mems_i, noise_exp = noise_exp)
    } else {
      policies[[i]] <- NA
      mems_i <-NA
    }
    mems[[i]] <- mems_i
  }
  
  df <- bind_rows(df_list)
  
  if(policy_exists) {
    policies <- bind_rows(policies)
    mems <- bind_rows(mems)
  }
  
  #more work to do with df
  return(list(df, policies, opt_policy, mems))
}

clean_df <- function(df, total_eps = 500) {
  df <- df %>% filter(!is.na(replicate), 
                            condition != "evodm" | ep_number > total_eps) %>%
    dplyr::mutate(episode = ifelse(ep_number>total_eps, ep_number-total_eps, ep_number)) %>%
    mutate(replicate = as.numeric(replicate),
           replicate_mod = as.numeric(replicate_mod),
           replicate = replicate + (replicate_mod-1), 
           replicate = ifelse(replicate > 100, replicate - 100, replicate))
  
  df <- df %>% 
    filter((train_input != "sv" | train_input == "sv" & condition == "evodm")) %>% 
    mutate(condition = ifelse(train_input == "sv", "evodm_sv", condition))
  
  df <- df  %>% 
    group_by(replicate, condition, reset, episode, quadmut, totalres) %>% 
    summarise(fitness = mean(average_fitness))
  return(df)
}

prep_results <- function(total_eps = 500, N = 4, landscapes = "mira", policy_exists = FALSE, dir = "test") {
  out <- prep_performance_overview(N=4, landscapes = "mira", policy_exists = FALSE, dir = "test")
  df <- clean_df(out[[1]],total_eps = total_eps) 
  return(df)
}


prep_mira_heatmap <- function(total_eps = 500, update = FALSE, noise_exp= FALSE, do_clustering=FALSE) {
  if(noise_exp) {
    if(file.exists("data/results/heatmap_raw_noise.Rda") & !update) {
      load("data/results/heatmap_raw_noise.Rda")
    } else {
      out <- prep_performance_overview(N=4, landscapes = "mira", noise_exp = TRUE)
      save(out, file = "data/results/heatmap_raw_noise.Rda")
    }
  } else {
    if(file.exists("data/results/heatmap_raw.Rda") & !update) {
      load("data/results/heatmap_raw.Rda")
    } else {
      out <- prep_performance_overview(N=4, landscapes = "mira")
      save(out, file = "data/results/heatmap_raw.Rda")
    }
  }
  
  #drug index table
  drug_index <- data.frame(action = 1:15, 
                           drug = c("AMP", "AM", "CEC", "CTX", "ZOX", "CXM", 
                                    "CRO", "AMC", "CAZ", "CTT", "SAM", "CPR",
                                    "CPD", "TZP", "FEP"))
  
  #first prep the performance of all 100 runs for the lollipop
  df <- out[[1]] %>% filter(!is.na(replicate), 
                            condition != "evodm" | ep_number > total_eps) %>%
    dplyr::mutate(episode = ifelse(ep_number>total_eps, ep_number-total_eps, ep_number)) %>%
    mutate(replicate = as.numeric(replicate),
           replicate_mod = as.numeric(replicate_mod),
           replicate = replicate + (replicate_mod-1), 
           replicate = ifelse(replicate > 100, replicate - 100, replicate)) # this one because at one point I had replicates 101-200 for no reason
  
  if(noise_exp) {
    while(max(df$replicate) > 10) {
      df <- df %>% mutate(replicate = ifelse(replicate > 10, replicate - 10, replicate))
    }
  }
  
  #set it up to include evodm_sv as a comparison group
  df <- df %>% 
    filter((train_input != "sv" | train_input == "sv" & condition == "evodm")) %>% 
    mutate(condition = ifelse(train_input == "sv", "evodm_sv", condition))
  
  df_sequences <- get_common_sequence(df)
  
  #bind all experimental conditions back together
  joint_probs_drug <- compute_joint_probability(df = df, num_drugs = 15)
  joint_probs_state <- compute_joint_probability(df = df, num_states =16, state = TRUE)
  df_orig <- df
  
  #finish cleaning performance data
  
  if(noise_exp) {
    df <- df  %>% 
      group_by(replicate, condition, reset, episode, quadmut, totalres, 
               noise_modifier) %>% 
      summarise(fitness = mean(average_fitness)) %>% 
      pivot_wider(names_from = "condition", values_from = "fitness") %>% 
      mutate(benefit = naive - evodm,
             distance_optimal = optimal_policy - evodm,
             optimal_benefit = naive - optimal_policy
      ) %>% 
      ungroup() 
    
    #need replicate to still be numeric at this stage
    df_small <- df %>% 
      filter(quadmut == FALSE, totalres == FALSE) %>% 
      group_by(replicate, reset, noise_modifier) %>% 
      summarise(benefit = mean(benefit))
  } else {
    df <- df  %>% 
      group_by(replicate, condition, reset, episode, quadmut, totalres) %>% 
      summarise(fitness = mean(average_fitness)) %>% 
      pivot_wider(names_from = "condition", values_from = "fitness") %>% 
      mutate(benefit = naive - evodm, 
             benefit_sv = naive - evodm_sv,
             distance_optimal = optimal_policy - evodm,
             optimal_benefit = naive - optimal_policy
      ) %>% 
      ungroup() 
    
    #need replicate to still be numeric at this stage
    df_small <- df %>% 
      filter(quadmut == FALSE, totalres == FALSE) %>% 
      group_by(replicate, reset) %>% 
      summarise(benefit = mean(benefit),
                benefit_sv = mean(benefit_sv))
  }
  
  
  #re-order to be in descending order of benefit 
  df <- df %>%
    mutate(
      replicate = factor(replicate),
      replicate = fct_reorder(replicate, benefit, .desc=TRUE)
    ) 
  
  #now grab data on just one replicate for a snapshot
  #use the middle replicate
  rep_target <- levels(df$replicate)[round(length(levels(df$replicate))/2)]
  
  df_full <- out[[1]] %>% 
    filter(replicate == rep_target)
  
  df_long <- df_full %>% 
    pivot_longer(cols = matches("\\d"), names_to = "action", 
                 values_to = "moving_prob") %>% 
    rename(drug_regime = drug) %>% 
    mutate(action = as.numeric(action)) %>%
    left_join(drug_index) %>% 
    filter(condition == "evodm")
  
  
  ##Process the policies data - many of the same steps as for the performance data
  policies <- out[[2]] %>% 
    filter(!is.na(replicate)) %>% 
    mutate(replicate = as.numeric(replicate),
           replicate_mod = as.numeric(replicate_mod),
           replicate = replicate + (replicate_mod-1),
           replicate = ifelse(replicate > 100, replicate - 100, replicate)) %>% # this one because at one point I had replicates 101-200 for no reason
    filter(episode == total_eps, !quadmut, !totalres) 
  
  #
  policies_state <- policies %>% 
    group_by(state, action, train_input) %>% 
    summarise(prob_selection = mean(prob_selection)) %>% 
    left_join(drug_index)
  
  #add drug index to optimal policy
  drug_index <- drug_index %>% mutate(action = action-1)
  opt_policy <- out[[3]] %>% 
    rename(time_step = action,
           action = optimal_action) %>% 
    left_join(drug_index) %>% 
    mutate(prob_selection = 1)
  
  #undo the bit we did earlier
  drug_index <- drug_index %>% mutate(action = action+1)
  if(!noise_exp & do_clustering) {
    #make mega drug index
    df_list <- list()
    counter = 0
    for (i in 1:length(unique(opt_policy$state))) {
      for(j in 1:20) {
        counter = counter+1
        df_ij <- drug_index
        df_ij$state = i
        df_ij$time_step=j
        df_list[[counter]] <- df_ij 
      }
    } 
    
    #Format optimal like we need and bind to other policies
    drug_index2 = bind_rows(df_list)
    opt_policy2 <- drug_index2 %>% left_join(opt_policy) %>% 
      mutate(prob_selection = ifelse(is.na(prob_selection), 0, prob_selection)) %>% 
      group_by(action, state) %>% 
      summarise(prob_selection = mean(prob_selection)) %>%
      mutate(train_input = "optimal",
             replicate = 201)
    
    policies2 <- bind_rows(policies, opt_policy2) %>% 
      mutate(replicate = ifelse(train_input == "sv", replicate + 100, replicate))
    
    clusters <- cluster_policy(policies2)
    policies_state_clust <- policies2 %>% 
      left_join(clusters[[1]]) %>%
      group_by(state, action, cluster) %>% 
      summarise(prob_selection = mean(prob_selection)) %>% 
      left_join(drug_index)
  } else {
    clusters <- list()
    policies_state_clust <- list()
  }
  
  
  #finish summarising the policy df
  policies <- policies %>% 
    group_by(action, replicate, reset, train_input) %>% 
    summarise(prob_selection = mean(prob_selection)) %>% 
    ungroup() %>% 
    left_join(drug_index) %>%
    left_join(df_small) %>% 
    mutate(replicate = factor(replicate),
           replicate = fct_reorder(replicate, benefit, .desc = TRUE))
  
  mems <- out[[4]] %>%
    mutate(replicate= as.numeric(replicate))
  
  out <- list(policies, df, df_long, df_full, opt_policy, joint_probs_drug, mems,
              policies_state, joint_probs_state, df_orig, clusters, policies_state_clust, df_sequences)
  
  return(out)
  #now compute 
}

cluster_trajectory <- function(df) {
  
}

#Function to cluster policies
cluster_policy <- function(df) {
  mat <-  df %>% select(state, action, replicate, prob_selection, train_input) %>%
    pivot_wider(id_cols = c(replicate, train_input), names_from = c(state, action), values_from = prob_selection) %>% 
    mutate(replicate = as.numeric(replicate)) %>%
    arrange(replicate )%>% 
    select(-train_input, -replicate) %>% as.matrix()
  
  mat[is.na(mat)] <- 0
  
  #Everything below here is going to have to happen twice so lets make a function
  
  pca_out <- prcomp(mat)
  
  df <- data.frame(
    replicate = 1:201, 
    condition = c(
      replicate(100, "RL-fit"),
      replicate(100, "RL-genotype"),
      "MDP"
    ),
    pc1 = pca_out$x[,1],
    pc2 = pca_out$x[,2],
    pc3 = pca_out$x[,3],
    pc4 = pca_out$x[,4],
    pc5 = pca_out$x[,5],
    pc6 = pca_out$x[,6],
    pc7 = pca_out$x[,7],
    pc8 = pca_out$x[,8],
    pc9 = pca_out$x[,9],
    pc10 = pca_out$x[,10],
    pc11 = pca_out$x[,11],
    pc12 = pca_out$x[,12],
    pc13 = pca_out$x[,13],
    pc14 = pca_out$x[,14],
    pc15 = pca_out$x[,15]
  )
  
  df <- df %>% filter(replicate != 201) #don't use the MDP replicate here in the end
  
  #fviz_nbclust(df[,3:ncol(df)], FUNcluster = kmeans)
  km.res <- kmeans(df[,3:ncol(df)], centers = 5)
  
  df$cluster <- km.res$cluster
  
  sum <- summary(pca_out)
  importance <- c(sum$importance[2], sum$importance[5], sum$importance[8])
  
  return(list(df, importance))
  
  
}
#function to compute the joint probability of a drug being selected in the 500 episode 
#going to co-opt this to compute the edge counts for every state pair
compute_joint_probability <- function(df, state= FALSE, num_drugs, num_states) {
  #this is going to be different for every reset
  #nutso looping structure here
  reset_vec <- as.numeric(unique(df$reset))
  condition_vec <- unique(df$condition)
  if(state) {
    action_vec <- 1:num_states
  } else {
    action_vec <- 1:num_drugs
  }
  
  
  #list for outer loop
  out_list <- list()
  iter = 1
  for(i in 1:length(reset_vec)) {
    for(c in condition_vec) {
      #Compute joint probability for every condition
      df_i <- df %>% filter(reset == reset_vec[i], condition == c, quadmut == FALSE, totalres == FALSE)
      #Collapse the drug sequence for a given episode into a string
      
      if(state) {
        df_i <- df_i %>% group_by(ep_number, replicate) %>% 
          summarise(var = paste(state, collapse = ",", sep = ","), count = n())
      } else {
        df_i <- df_i %>% group_by(ep_number, replicate) %>% 
          summarise(var = paste(drug, collapse = ",", sep = ","))
      }
      
      
      out_inner <- list()
      for(j in 1:length(action_vec)) {
        
        #Extract the actions that were preceded by the action under study. 
        str_pattern = paste0("(?<=,", action_vec[j], "),\\d+")
        out = str_extract_all(df_i$var, pattern = str_pattern)
        #setup counter
        #oh my god another loop help me
        if(state) {
          event_counts = replicate(num_states, 0)
          drug2 = seq(1, num_states,by=1)
        } else {
          event_counts = replicate(num_drugs, 0)
          drug2 = seq(1, num_drugs,by=1)
        }
        
        for(z in 1:length(out)) {
          events <- as.numeric(gsub(",", "", out[[z]]))
          for(y in 1:length(events)) { #5 loops baby
            event_counts[events[y]] =  event_counts[events[y]] + 1 #count all the times action b followed action a
          }
        }
        
        #get number of total events so we can compute the joint prob
        num_events = sum(event_counts)
        df_j <- data.frame(drug1 = action_vec[j], drug2 = drug2, event_counts = event_counts, 
                           joint_prob = event_counts/num_events)
        out_inner[[j]] <- df_j
        
        
      }
      
      #clean and store results of inner loop
      df_i <- bind_rows(out_inner)
      df_i$reset <- reset_vec[[i]]
      df_i$condition <- c
      out_list[[iter]] <- df_i
      iter = iter + 1 #increment counter by 1
    }
  }
  #clean and process results of two loops
  df <- bind_rows(out_list)
  return(df)
}

compute_combo_graph <- function(joint_probs, reset = 20, min_prob = 0, min_degree=0, state){
  
  drug_index <- data.frame(name = 1:15, 
                           label = c("AMP", "AM", "CEC", "CTX", "ZOX", "CXM", 
                                     "CRO", "AMC", "CAZ", "CTT", "SAM", "CPR",
                                     "CPD", "TZP", "FEP"),
                           label_long = c("Ampicillin", "Amoxicillin", "Cefaclor", "Cefotaxime",
                                          "Ceftizoxime", "Cefuroxime", "Ceftriaxone",
                                          "Amoxicillin + Clavulanic acid", "Ceftazidime",
                                          "Cefotetan", "Ampicillin + Sulbactam", 
                                          "Cefprozil", "Cefpodoxime", 
                                          "Pipercillin + Tazobactam", "Cefepime")
  )
  
  #Filter joint_probs to only have the reset distance under study. 
  df_i <- filter(joint_probs, reset == reset)
  
  #put together an adjacency matrix
  edge_mat <- df_i %>% select(drug1, drug2, joint_prob) %>% 
    mutate(joint_prob = ifelse(joint_prob < min_prob, 0, joint_prob)) %>%
    pivot_wider(names_from = drug2, values_from = joint_prob) %>% 
    arrange(-desc(drug1)) %>% 
    select(-drug1) %>%
    as.matrix()
  
  g <- igraph::graph_from_adjacency_matrix(edge_mat, weighted=  TRUE, mode = "directed")
  E(g)$width <- E(g)$weight
  
  
  g <- as_tbl_graph(g) %>%
    activate(nodes) %>%
    mutate(degree = centrality_degree(), 
           name = as.integer(name)) %>% 
    left_join(drug_index) %>%
    filter(degree >= min_degree)
  
  if(state){
    g <- g %>%
      activate(nodes) %>% 
      mutate(label = as.character(name))
  }
  
  return(g)
}

#function to compute a graph where edge weights correspond to the probability of one state following another
prep_lr <- function(lr_range = 10^seq(1,3,by= 0.2)/1e7) {
  #  #drug index table
  
  out <- prep_performance_overview(N=4, landscapes = "lr", lr_range = lr_range)
  
  #first prep the performance of all 70 runs for the lollipop
  mems <- out[[4]] %>% 
    group_by(lr, original_ep) %>% 
    summarise(fitness = mean(fitness))
  
  return(mems)
}

prep_update_target <- function() {
  out <- prep_performance_overview(N=4, landscapes = "updatetarget", lr_range = NULL)
  
  #first prep the performance of all 70 runs for the lollipop
  mems <- out[[4]] %>% 
    group_by(update_target, original_ep) %>% 
    summarise(fitness = mean(fitness))
  return(mems)
}

prep_batch_size <- function() {
  out <- prep_performance_overview(N=4, landscapes = "batchsize", lr_range = NULL)
  
  #first prep the performance of all 70 runs for the lollipop
  mems <- out[[4]] %>% 
    group_by(batch_size, original_ep) %>% 
    summarise(fitness = mean(fitness))
  return(mems)
}

prep_gamma <- function() {
  
  out <- prep_performance_overview(N=4, landscapes = "gamma", lr_range = NULL)
  
  df <- out[[1]] %>% filter(!is.na(replicate), ep_number %in% 450:500) %>%
    group_by(replicate, gamma, condition) %>% 
    summarise(fitness = mean(average_fitness)) %>% 
    pivot_wider(names_from = "condition", values_from = "fitness") %>% 
    mutate(benefit = naive - evodm,
           distance_optimal = optimal_policy - evodm, 
           replicate = factor(replicate)) %>% 
    ungroup() %>%
    mutate(replicate = fct_reorder(replicate, benefit, .desc=TRUE)) 
  return(df)
  
}

prep_opt_policy <- function(df) {
  drug_index <- data.frame(action = 1:15, 
                           drug = c("AMP", "AM", "CEC", "CTX", "ZOX", "CXM", 
                                    "CRO", "AMC", "CAZ", "CTT", "SAM", "CPR",
                                    "CPD", "TZP", "FEP"))
  df_opt <- df %>% filter(condition == "optimal_policy") 
  
  #Put together figure showing probability of a drug being selected over the course of an episode (for the optimal)
  opt_freq_table <- (table(df_opt$drug, df_opt$action_number))
  opt_props <- as.data.frame(opt_freq_table/Matrix::colSums(opt_freq_table)) %>%
    rename(action = Var1, time_step = Var2, prob_selection = Freq) %>% 
    mutate(action = as.numeric(as.character(action)),
           time_step = as.numeric(time_step),
           prob_selection = as.numeric(prob_selection)) %>% # have to add an as.character for some reason
    left_join(drug_index)
  
  opt_props_sum <- opt_props %>% 
    group_by(action, drug) %>% 
    summarise(prob_selection = mean(prob_selection)) 
  
  opt_props_sum <- left_join(drug_index, opt_props_sum) %>% 
    mutate(prob_selection = ifelse(is.na(prob_selection), 0, prob_selection))
  
  return(list(opt_props, opt_props_sum))
}

prep_twodrug_sweep <- function(update = FALSE) {
  if(file.exists("data/results/twodrugsweep.Rda") & !update) {
    load("data/results/twodrugsweep.Rda")
    return(out)
  }
  drug_index1 <- data.frame(drug1 = 0:14, 
                            drug_name1 = c("AMP", "AM", "CEC", "CTX", "ZOX", "CXM", 
                                           "CRO", "AMC", "CAZ", "CTT", "SAM", "CPR",
                                           "CPD", "TZP", "FEP"))
  drug_index2 <- data.frame(drug2 = 0:14, 
                            drug_name2 = c("AMP", "AM", "CEC", "CTX", "ZOX", "CXM", 
                                           "CRO", "AMC", "CAZ", "CTT", "SAM", "CPR",
                                           "CPD", "TZP", "FEP"))
  #run function to sweep through all two drug policies and compute performance over 100 episodes
  out <- policy_sweep(episodes = as.integer(100))
  #loop through output list
  out2 <- list()
  for(i in 1:length(out)) {
    #clean mem for every policy
    mem_i <- clean_mem(out[[i]][[1]])
    
    #add which drugs made up the alternating two drug policy
    combo_i <- unlist(out[[i]][[2]])
    mem_i$drug1 <- combo_i[1]
    mem_i$drug2 <- combo_i[2]
    mem_i$starting_state <- out[[i]][[3]]
    
    #done
    out2[[i]] <- mem_i
  }
  
  #get landscape correlations
  df <- compute_ls_cor()
  
  df_long <- bind_rows(out2) %>% 
    left_join(df) %>% 
    left_join(drug_index1) %>% left_join(drug_index2) %>% 
    group_by(drug_name1, drug_name2, ep_number, starting_state) %>% 
    summarise(average_fitness = mean(average_fitness), correl = mean(correl)) %>%
    ungroup() %>%
    mutate(combo =  paste(drug_name1, drug_name2, sep = ",")) 
  
  df_hist <- df_long %>% group_by(drug_name1, drug_name2, combo, starting_state) %>% 
    summarise(average_fitness = mean(average_fitness), 
              correl = mean(correl))
  
  #order fct
  df_long <- df_long %>% 
    mutate(combo = fct_reorder(combo, average_fitness)) %>%
    filter(starting_state == 0)
  
  df_hist_wt <- df_hist %>% 
    filter(starting_state == 0)
  
  out <- list(df_long, df_hist, df_hist_wt)
  save(out, file = "data/results/twodrugsweep.Rda")
  return(out)
  
}


## Function to compute pairwise correlation of each mira landscape
#out: df with pairwise mira correlations

compute_ls_cor <- function(){
  drugs <- define_mira_landscapes() 
  
  #get ready to loop through
  drug1_vec <- vector(mode = "numeric", length = length(drugs)^2)
  drug2_vec <- vector(mode = "numeric", length = length(drugs)^2)
  cor_vec <- drug1_vec
  
  counter = 0
  for(i in 1:length(drugs)) {
    for(j in 1:length(drugs)) {
      counter = counter+1
      cor_vec[counter] <- cor(drugs[[i]], drugs[[j]], method = "pearson")
      drug1_vec[counter] <- i
      drug2_vec[counter] <- j
    }
  }
  df <- data.frame(drug1 = drug1_vec-1, drug2 = drug2_vec-1, correl = cor_vec)
  return(df)
  
  
}

prep_network <- function(joint_prob) {
  drug_index <- data.frame(id = 1:15, 
                           label = c("AMP", "AM", "CEC", "CTX", "ZOX", "CXM", 
                                     "CRO", "AMC", "CAZ", "CTT", "SAM", "CPR",
                                     "CPD", "TZP", "FEP"),
                           label_long = c("Ampicillin", "Amoxicillin", "Cefaclor", "Cefotaxime",
                                          "Ceftizoxime", "Cefuroxime", "Ceftriaxone",
                                          "Amoxicillin + Clavulanic acid", "Ceftazidime",
                                          "Cefotetan", "Ampicillin + Sulbactam", 
                                          "Cefprozil", "Cefpodoxime", 
                                          "Pipercillin + Tazobactam", "Cefepime")
  )
  edges <- joint_prob %>% 
    rename(from = drug1, to = drug2, value = event_counts) %>% filter(value > 2000) %>% 
    mutate(value = value/10)
  nodes <- data.frame(id = unique(c(edges$from, edges$to))) %>% 
    left_join(drug_index)
  return(list(nodes, edges))
  
}

s2n<- function() {
  noise_vec <- seq(0,100, by = 1)
  df <- signal2noise(noise_vec)
  df <- df %>% 
    mutate(s2nr = fitness / (fitness + abs(fitness - noisy_fitness))) %>% 
    group_by(noise_modifier) %>% 
    summarise(s2nr = mean(s2nr), fitness = mean(fitness), noisy_fitness = mean(noisy_fitness))
  return(df)
}

get_common_sequence <- function(out) {
  drug_index <- data.frame(drug = 1:15, 
                           drug_code = c("AMP", "AM", "CEC", "CTX", "ZOX", "CXM", 
                                         "CRO", "AMC", "CAZ", "CTT", "SAM", "CPR",
                                         "CPD", "TZP", "FEP"))
  df <- out %>%
    filter(quadmut==FALSE, 
           totalres==FALSE) %>% 
    distinct(ep_number, action_number, train_input, replicate, .keep_all = TRUE) %>%
    left_join(drug_index, by = 'drug')%>%
    group_by(ep_number, replicate, train_input) %>% 
    summarise(drug_sequence = paste(drug_code, collapse = ",")) %>% 
    ungroup() %>% 
    group_by(drug_sequence, replicate, train_input) %>% 
    summarise(n = n()) %>%
    ungroup() %>% 
    group_by(replicate, train_input) %>%
    slice_max(n=1, order_by = n, with_ties = FALSE)
  return(df)
  
}

define_landscape_graph <- function() {
  opp_ls <- compute_opp_ls(c("CTX", "AMP", "CPR", "SAM", "TZP"))
  opp_landscape <- Landscape(ls = opp_ls, N=as.integer(4), sigma=0.5)
  tm = opp_landscape$get_TM()
  tm = t(tm)
  g <- igraph::graph_from_adjacency_matrix(tm, weighted = TRUE)
  g <- set_vertex_attr(g, name = "val", value = unlist(opp_ls))
  g <- as_tbl_graph(g)
  
  return(g)
}

