#Always load the python stuff in a clean session\
#
#reticulate::use_python("C:/Program Files/Python37/python.exe", required = TRUE) #python path may vary

reticulate::source_python("load.py")

#packages
library(here)
library(stringr)
library(magrittr)
library(tidyr)
library(stringr)
library(dplyr)
library(forcats)
library(ggplot2)
library(cowplot)
library(ggbeeswarm)
library(ggpubr)
library(visNetwork)
library(scales)
library(ggraph)
library(igraph)
library(tidygraph)


#Helper functions
source(here("R", "prep_figs.R"))
source(here("R", "clean_evol_out.R"))
#set theme
theme_set(theme_classic())

out_noise <- prep_mira_heatmap(total_eps=500, update=FALSE, noise_exp = TRUE)
##Load/ clean data
##Heatmap describing repeatability of results
out_hm <- prep_mira_heatmap(total_eps = 500, update =FALSE, do_clustering=FALSE)

out_combos <- prep_twodrug_sweep()
df_combos_long <- out_combos[[1]]
df_combos <- out_combos[[3]]
df_combos_big <- out_combos[[2]]



#Do we want a different intro figure
#I actually think this belongs in a different plot
# g3 <- ggplot(df_big_wide, aes(x = index_main, y = fitness_ma, color = condition)) +
#   geom_line() + 
#   labs(x = "Evolutionary time", 
#        y = "Population fitness \n (150 step moving average)", 
#        tag = "B") +
#   theme(text = element_text(size = 16))

##############Mira Performance Plot####################
#All code to generate mira performance plot
mira_perf_plot <- function(out, out_noise, df_combos_long, df_combos, reset_every = 20, 
                           replicates = 75, quad_mut = FALSE, total_res = FALSE) {
  
  mem_df <- out[[7]] %>% 
    filter(replicate <= replicates,
           quadmut == quad_mut,
           totalres == totalres)
  df_big <- out[[3]]
  df_byaction <- out[[10]] %>% 
    group_by(replicate, condition, reset,  quadmut, totalres, action_number) %>% 
    summarise(fitness = mean(average_fitness)) %>% 
    ungroup() 
  
  df_noise <- out_noise[[2]]
  df_noise_sum <- s2n()
  
  df <- out[[2]] %>% 
    filter(reset == reset_every, replicate %in% 1:replicates,
           quadmut == quad_mut,
           totalres == total_res)
  
  mem_short <- mem_df %>% group_by(original_ep, train_input) %>% 
    summarise(
      upper_bound = quantile(fitness, probs = c(0.95)), 
      lower_bound = quantile(fitness, probs = c(0.05)),
      fitness = mean(fitness)
    )
  
  g1 <- ggplot(mem_short, aes(x = original_ep, y = fitness, color = train_input)) + 
    geom_line() + 
    geom_ribbon(aes(ymin = lower_bound, ymax = upper_bound), alpha = 0.3) + 
    theme(text = element_text(size = 16)) + 
    scale_color_discrete(labels = c("fitness", "genotype")) + 
    labs(x = "original episode", y = "population fitness", tag = "A",  
         color = "training \ninput")
  
  g2 <- ggplot(df_noise, aes(x = as.numeric(noise_modifier), y = benefit, group = noise_modifier)) + 
    geom_boxplot() + 
    theme(text = element_text(size=16)) + 
    geom_hline(aes(yintercept = 0), linetype = "dotted") + 
    labs(x = "noise modifier", x = "fitness benefit (RL_fit - random)",
         tags = "B")
  
  # g4 <- ggplot(df_noise_sum, aes(x = factor(noise_modifier), y = s2nr)) + 
  #   geom_segment(aes(x = factor(noise_modifier), 
  #                    xend = factor(noise_modifier), y = 0, yend = s2nr)) + 
  #   geom_point() + 
  #   labs(x = "Noise Modifier", y = "Signal to Noise Ratio", tag = "D") + 
  #   theme(text = element_text(size=16))
  
  g4 <- ggplot(df_noise_sum, aes(x = noise_modifier, y = s2nr)) + 
    geom_segment(aes(x = noise_modifier, 
                     xend = noise_modifier, y = 0, yend = s2nr)) + 
    geom_point() + 
    labs(x = "Noise Modifier", y = "Signal to Noise Ratio", tag = "D") + 
    theme(text = element_text(size=16))
  
  
  #now histogram 
  df_hist <- df %>% 
    pivot_longer(cols = c(evodm, naive, optimal_policy, evodm_sv), 
                 names_to =  "condition", 
                 values_to = "population_fitness") 
  
  g3 <- ggplot(df_hist, aes(x = population_fitness, fill = condition)) + 
    geom_density(alpha = 0.6) + 
    geom_density(data = df_combos, aes(x=average_fitness, 
                                       fill = "two-drug \n combinations"), alpha = 0.6) + 
    scale_fill_viridis_d(labels = c("RL_fit", "RL_genotype", "random", "MDP", "two-drug \n combinations")) + 
    scale_x_continuous(expand = c(0,0), limits = c(0,3.3)) + 
    scale_y_continuous(expand = c(0,0)) + 
    labs(y = "density", x = "population fitness", tags = "C") + 
    theme(text = element_text(size = 16),
          legend.position = "top", 
          legend.title = element_blank())
  
  g <- plot_grid(g1, g2, g3, g4, nrow = 2, rel_heights = c(1.25,1), align = "vh", axis = "blr")
  
  return(g)
  
}

mira_perf_supps <- function(out) {
  df <- out[[2]] %>% 
    filter(reset == 20, replicate %in% 1:100,
           quadmut == FALSE,
           totalres == FALSE) %>% 
    mutate(sv_decrement = optimal_policy - evodm_sv)
  
  df_byaction <- out[[10]] %>% 
    group_by(replicate, condition, reset,  quadmut, totalres, action_number) %>% 
    summarise(fitness = mean(average_fitness)) %>% 
    ungroup() 
  
  g1 <- ggplot(df, aes(x = replicate, y = benefit)) +
    geom_boxplot() +
    xlab("Replicate") +
    ylab("Fitness improvement \n (random - RL_fit)") +
    ggtitle("RL-Fit performance relative to the random \ndrug cycling condition for all 100 replicates") + 
    labs(tag = "A") +
    theme(axis.text.x = element_blank(),
          text = element_text(size = 16),
          plot.margin = unit(c(0,0,0,0), "cm")) + 
    geom_hline(aes(yintercept = 0), linetype = "dotted")
  
  g2 <- ggplot(df, aes(x = replicate, y = distance_optimal)) +
    geom_boxplot() +
    xlab("Replicate") +
    ylab("Fitness decrement \n (optimal - RL_fit)") +
    labs(tag = "B") +
    ggtitle("RL-Fit performance relative to the optimal \ndrug cycling condition for all 100 replicates") + 
    theme(axis.text.x = element_blank(),
          text = element_text(size = 16),
          plot.margin = unit(c(0,0,0,0), "cm")) + 
    geom_hline(aes(yintercept = 0), linetype = "dotted")
  
  g3 <- ggplot(df, aes(x = replicate, y = benefit_sv)) +
    geom_boxplot() +
    xlab("Replicate") +
    ylab("Fitness improvement \n (random - RL_genotype)") +
    labs(tag = "C") +
    ggtitle("RL-genotype performance relative to the random \ndrug cycling condition for all 100 replicates") + 
    theme(axis.text.x = element_blank(),
          text = element_text(size = 16),
          plot.margin = unit(c(0,0,0,0), "cm")) + 
    geom_hline(aes(yintercept = 0), linetype = "dotted")
  
  g4 <- ggplot(df, aes(x = replicate, y = sv_decrement)) + 
    geom_boxplot() +
    xlab("Replicate") +
    ylab("Fitness decrement \n (optimal - RL_genotype)") +
    ggtitle("RL-genotype performance relative to the optimal \ndrug cycling condition for all 100 replicates") + 
    labs(tag = "D") +
    theme(axis.text.x = element_blank(),
          text = element_text(size = 16),
          plot.margin = unit(c(0,0,0,0), "cm")) + 
    geom_hline(aes(yintercept = 0), linetype = "dotted")
  #g3 <- ggplot(df_byaction, aes(x = factor(action_number), y=  fitness, 
  #                              color = condition)) + 
  #  geom_violin()
  
  plot_grid(g1,g2, g3, g4)
}

mira_perf_sens <- function(out) {
  mem_df <- out[[7]] %>% 
    mutate(
      condition = case_when(
        quadmut ~ "quadmut",
        totalres ~ "totalres",
        TRUE ~ "base_case")
    ) %>% 
    filter(original_ep < 500)
  
  mem_short <- mem_df %>% group_by(original_ep, condition) %>% 
    summarise(
      upper_bound = quantile(fitness, probs = c(0.95)), 
      lower_bound = quantile(fitness, probs = c(0.05)),
      fitness = mean(fitness)
    )
  
  g1 <- ggplot(mem_short, aes(x = original_ep, y = fitness, color = condition)) + 
    geom_line() + 
    geom_ribbon(aes(ymin = lower_bound, ymax = upper_bound), alpha = 0.3) + 
    theme(text = element_text(size = 16)) + 
    labs(x = "original episode", y = "population fitness", tag = "A")
  
  return(g1)
  
}
###############Mira Policy Plot#################
mira_policy_plot <- function(out, cluster_bool = FALSE) {
  #clusters <- out[[11]][[1]]
  policies_state_clust <- out[[12]]
  importance <- out[[11]][[2]]
  df_big_wide <- out[[4]]
  
  policies <- out[[1]] %>% filter(reset == 20) %>% 
    mutate(benefit = ifelse(train_input == "sv", benefit_sv, benefit),
           replicate = fct_reorder(replicate, benefit, .desc=TRUE), 
           replicate_num = as.numeric(replicate),
           replicate_num = ifelse(train_input == "fit", replicate_num + 100, replicate_num)
    )
  
  #grab a drug sequence from the best performing replicate
  top_rep <- levels(policies$replicate)[1:3]
  sequence <- out[[13]] %>% filter(replicate %in% top_rep)
  
  df_perf <- policies %>% group_by(replicate_num) %>%
    summarise(benefit = median(benefit))
  
  policies_state =  out[[8]] %>% 
    mutate(train_input = ifelse(train_input == "fit", "fitness", "genotype"))
  
  out <- prep_opt_policy(df_big_wide)
  opt_props <- out[[1]]
  opt_props_sum <- out[[2]]
  opt_props_sum$replicate_num = -1
  
  g1a <- ggplot(policies, aes(x = replicate_num, y = factor(drug), fill = prob_selection)) + 
    geom_tile() + 
    scale_fill_gradient(low = "navy", high = "yellow", 
                        labels = number_format(accuracy =0.1)) + 
    scale_x_continuous(expand = c(0,0), limits = c(-5,200)) + 
    scale_y_discrete(expand = c(0,0)) + 
    labs(y = "drug", x = "", tag = "A", 
         fill = "Probability of \n selection") + 
    theme(axis.text.x = element_blank(), 
          axis.ticks.x = element_blank(),
          axis.title.x = element_blank(),
          text = element_text(size = 16),
          legend.margin = margin(), 
          legend.key.width = unit(1, units = "cm"),
          legend.position = "none",
          plot.margin = unit(c(0.2,0.2,5.5,0.2), "cm")) + 
    geom_vline(xintercept = 100, color = 'white')
  
  g_opt <- ggplot(opt_props_sum, aes(x = replicate_num, y = factor(drug), fill = prob_selection)) + 
    geom_tile() + 
    scale_fill_gradient(low = "navy", high = "yellow") + 
    theme(axis.line = element_blank(),
          axis.text = element_blank(), 
          legend.position = "none",
          axis.ticks = element_blank(), 
          axis.title = element_blank(), 
          margin()) + 
    scale_y_discrete(expand = c(0,0)) + 
    scale_x_continuous(expand = c(0,0)) + theme_nothing()
  
  g_perf <- ggplot(df_perf, aes(x = replicate_num, y = 0.5, fill = benefit)) + 
    geom_tile() + 
    scale_fill_viridis() + 
    theme(text = element_text(size=14), legend.position = "bottom",
          legend.key.width = unit(2.5, "cm"),
          axis.line.y = element_blank(),
          axis.text.y = element_blank(),
          axis.ticks.y = element_blank(),
          axis.title.y = element_blank(), 
          axis.text.x = element_blank(),
          plot.margin = unit(c(0,0,0,0), "cm"),
          legend.margin= margin(t=0,r=0,b=0,l=0)) +
    scale_x_continuous(expand = c(0,0), limits = c(0,200)) + 
    scale_y_continuous(expand = c(0,0)) + 
    labs(fill = 'benefit (RL - random)', x = "replicate") + 
    guides(fill = guide_colorbar(title.position = "top",
                                 title.hjust = unit(-0, "cm")))
  
  g1 <- g1a + 
    annotation_custom(ggplotGrob(g_opt), xmin = -5, xmax = 0, ymin = 0.5, ymax = 15.5) + 
    annotation_custom(ggplotGrob(g_perf), xmin = 0, xmax = 200, ymin = -5, ymax = 0.43)
  
  #now take a look at action by state heatmap
  g3 <- ggplot(policies_state, aes(x = drug, y = state, fill = prob_selection)) + 
    geom_tile() + 
    scale_fill_gradient(low = "navy", high = "yellow",
                        labels = number_format(accuracy =0.1)) + 
    facet_wrap(~train_input, nrow=2) + 
    theme(text = element_text(size = 16),
          legend.key.width = unit(1, units = "cm"),
          axis.text.x = element_text(angle = -60, vjust = 0.5, hjust = 0),
          plot.margin = unit(c(0,0,0,0), "cm")) + 
    labs(fill = "Probability of \n selection", tag = "B") + 
    scale_y_continuous(expand = c(0,0))
  
  if(cluster_bool) {
    g4 <- ggplot(clusters, aes(x = pc1, y = pc2, shape = condition, color = factor(cluster))) + 
      geom_point() + 
      theme(text = element_text(size = 16)) + 
      labs(x = paste0("PC1 (", importance[1]*100, "% of variance)"),
           y = paste0("PC2 (", importance[2]*100, "% of variance)"),
           color = "cluster")
    
    g5 <- ggplot(clusters, aes(x = pc1, y = pc3, shape = condition, color = factor(cluster))) + 
      geom_point() + 
      theme(text = element_text(size = 16)) + 
      labs(x = paste0("PC1 (", importance[1]*100, "% of variance)"),
           y = paste0("PC3 (", importance[3]*100, "% of variance)"),
           color = "cluster")
    
    g6 <- ggplot(clusters, aes(x = pc2, y = pc3, shape = condition, color = factor(cluster))) + 
      geom_point() + 
      theme(text = element_text(size = 16)) + 
      labs(x = paste0("PC2 (", importance[2]*100, "% of variance)"),
           y = paste0("PC3 (", importance[3]*100, "% of variance)"),
           color = "cluster")
    
    
    ### Prep a policy by cluster heatmap (no MDP)
    g7 <- ggplot(policies_state_clust %>% filter(!is.na(drug)), aes(x = drug, y = state, fill = prob_selection)) + 
      geom_tile() + 
      scale_fill_gradient(low = "navy", high = "yellow",
                          labels = number_format(accuracy =0.1)) + 
      facet_wrap(~cluster, nrow=2) + 
      theme(text = element_text(size = 16),
            legend.key.width = unit(1, units = "cm"),
            axis.text.x = element_text(angle = -60, vjust = 0.5, hjust = 0),
            plot.margin = unit(c(0,0,0,0), "cm")) + 
      labs(fill = "Probability of \n selection", tag = "B") + 
      scale_y_continuous(expand = c(0,0))
    
    g_pca <- plot_grid(g4,g5,g6)
    g <- list(g_pca, g7)
    
  } else {
    g <-plot_grid(g1,g3, ncol = 2, align = "hv", axis = "bt", rel_widths = c(1.75,1))
  }
  
  return(g)
  
}

landscape_summary <- function(out) {
  #grab state by action 
  policies_state =  out[[8]] %>% 
    mutate(train_input = ifelse(train_input == "fit", "fitness", "genotype")) %>% 
    filter(train_input == "genotype") %>% 
    group_by(state) %>% 
    slice_max(n=1, order_by = prob_selection, with_ties = FALSE)
  
  drugs <- define_mira_landscapes(as_dict=TRUE)
  
  num_winners <- vector(mode = "integer", length = length(drugs))
  num_losers <- vector(mode = "integer", length = length(drugs))
  max_fit <- vector(mode = "integer", length = length(drugs))
  min_fit <- vector(mode = "integer", length = length(drugs))
  out_loop <- list()
  for (i in 1:length(drugs)) {
    drug_i <- drugs[[i]]
    df_i <- data.frame(fitness = drug_i, state = 1:16, drug  = names(drugs)[i])
    out_loop[[i]] <- df_i
    num_winners[i] <- sum(drug_i <=1)
    num_losers[i] <- sum(drug_i > 1)
    max_fit[i] <- max(drug_i)
    min_fit[i] <- min(drug_i)
  }
  df <- data.frame(drug = names(drugs),
                   num_winners = num_winners,
                   num_losers = num_losers,
                   max_fit = max_fit,
                   min_fit = min_fit)
  #clean up 
  df2 <- bind_rows(out_loop)
  policies_state <- policies_state %>% left_join(df2)
  
  #define MDP policy as well
  opt_policy = out[[5]] %>% 
    group_by(state) %>% 
    filter(time_step==1) %>% 
    summarise(prob_selection = mean(prob_selection), 
              drug = unique(drug)) %>% 
    left_join(df2)
  
  #add bools for if the drug is used by RL-genotype or MDPFALSE
  df <- df %>% mutate(
    condition = ifelse(drug %in% unique(policies_state$drug), "blue"," black"),
    condition = ifelse(drug %in% unique(opt_policy$drug), "lightgreen", condition),
    condition = ifelse(drug %in% unique(opt_policy$drug) & drug %in% unique(policies_state$drug), "orange", condition),
    drug = fct_reorder(.f = drug, .x = num_winners, .desc = TRUE)) #
  
  color_vec <- df$condition
  names(color_vec) <- df$drug
  #re-order to be the same as the factor order
  color_vec <- color_vec[levels(df$drug)]
  #for whatever reason, the ties are fucked in the factoring so...
  
  df <- df %>% 
    pivot_longer(cols = c(num_winners, num_losers), names_to= "metric", values_to = "value")
  
  #add a group variable so we can color by what policy relies on what drug in which state
  policies_state <- policies_state %>% select(state, drug) %>% 
    mutate(grp = "RL-genotype")
  opt_policy <- opt_policy %>% select(state,drug) %>% 
    mutate(grp = "MDP")
  df2 <- df2 %>% left_join(opt_policy) %>% left_join(policies_state, by = c("state", "drug")) 
  df2 <- df2 %>% 
    mutate(grp = ifelse(is.na(grp.y), grp.x, grp.y),
           grp = ifelse(is.na(grp), "Not selected", grp))
  #plot everything, color coding based on what is selected by RL-genotype vs MDP
  g1 <- ggplot(df2, aes(x = state, y = fitness, group=state, color = grp)) +
    geom_boxplot() + 
    geom_quasirandom(size=1.75) + 
    theme(text = element_text(size = 14), 
          legend.title = element_blank()) + 
    labs(x = 'genotype', 
         tag = "C") + 
    scale_color_manual(values = c("orange", "black", "lightgreen"))
  
  g2 <- ggplot(df, aes(x = drug, y = value, fill = metric)) + 
    geom_col() + 
    labs(fill= "", 
         y = "Number of Genotypes",
         tag = "D") + 
    scale_fill_discrete(labels = c("Fitness > 1", "Fitness < 1")) + 
    scale_y_continuous(expand = c(0,0)) + 
    theme(text = element_text(size=14),
          axis.text.x = element_text(color = color_vec, 
                                     angle = -45, hjust = -0.2))
  
  g <- plot_grid(g1,g2, ncol = 2)
  return(g)
  
}
mira_traversal_plot<- function(out, mdp_bool = FALSE) {
  plot_graph <- function(g, legend = "none") {
    gg <- ggraph(g, layout = 'manual', x=x, y=y) + 
      geom_node_point(aes(fill=val), size = 5.5, shape = 21, color = "black") + 
      geom_edge_fan(aes(alpha = weight), strength = 2, edge_width = unit(1, "cm"),
                    arrow = arrow(ends = "last", length = unit(0.2, "cm"), angle = 30, type = "open"), 
                    show.legend = FALSE,
                    label_dodge = unit(-8, 'mm'),
                    label_size = 7,
                    end_cap = circle(4, 'mm')) +
      scale_edge_width(range = c(0.2, 4)) + 
      #geom_node_text(aes(label = label), size = 3, color = "white") + 
      scale_fill_gradient2(low = "black", mid = "white", high = "yellow") +
      theme(text = element_text(size = 9), legend.position = legend) + 
      labs(fill = "state \n value")
    return(gg)
  }
  #bring in all the data we need
  load("data/results/val_mat.Rda")
  val_mat <- val_mat[,1:20]
  val_df <- data.frame(name = 1:16, val = scale(rowMeans(val_mat))) %>%
    mutate(decrement = -val) %>% as_tibble()
  joint_prob = out[[9]]
  #first get the joint counts to compute the differences
  mdp_only <- joint_prob %>% filter(condition == "optimal_policy") %>% rename(joint_prob_mdp = 'joint_prob') %>% select(-event_counts, -condition)
  #get thhe joint_prob_mdp unmessed with to bind later
  mdp_only2 <- joint_prob %>% filter(condition == "optimal_policy") %>% mutate(condition = "mdp")
  #do some cleaning and compute the differences
  joint_prob2 <- joint_prob %>% 
    mutate(condition = 
             case_when(condition == "evodm" ~ 'RL-fit',
                       condition == "evodm_sv" ~ 'RL-genotype',
                       condition == "naive" ~ 'random',
                       condition == "optimal_policy" ~'mdp')
    ) %>% filter(condition != "random" & condition != "mdp") %>% 
    left_join(mdp_only) %>% 
    mutate(joint_prob = abs(joint_prob - joint_prob_mdp)) %>% 
    bind_rows(mdp_only2)
  
  g1 <- ggplot(joint_prob2, aes(x = factor(drug1), y = factor(drug2), fill = joint_prob)) +
    geom_tile() + 
    facet_wrap(~condition, nrow=1) + 
    scale_fill_viridis() + 
    labs(x = "state at time step t", y = "state at time step t + 1") + 
    scale_x_discrete(expand =c(0,0)) + 
    scale_y_discrete(expand = c(0,0)) +
    theme(text = element_text(size = 10))
  
  #specify node positions
  x = c(3.25, 0.5, 2.25, 0, 4, 1.25,2.5,0.5,5.75,3.75,5,2.25,6.25,4,5.75,3.25)
  y = c(1,      3,    3, 5, 3,    5,  5,  7,   3,   5,5,   7,   5,7,   7,  9)
  y = rev(y)
  joint_prob_sv <- joint_prob2 %>% filter(condition =="RL-genotype")
  g <- compute_combo_graph(joint_probs = joint_prob_sv, min_degree=0, min_prob = 0.05, state=TRUE)
  g <- g %>% 
    activate("nodes") %>%
    left_join(val_df)
  
  g2a <- plot_graph(g, legend = "right")
  
  ####
  joint_prob_fit <- joint_prob2 %>% filter(condition=="RL-fit")
  g <- compute_combo_graph(joint_probs = joint_prob_fit,min_degree=0, min_prob = 0.05, state = TRUE)
  g <- g %>% 
    activate("nodes") %>%
    left_join(val_df)
  g2b <- plot_graph(g, legend = "none")
  
  
  joint_prob_mdp <- joint_prob2 %>% filter(condition == "mdp")
  g <- compute_combo_graph(joint_probs = joint_prob_mdp, min_degree=0, min_prob = 0, state = TRUE)
  g <- g %>% 
    activate("nodes") %>%
    left_join(val_df)
  g2c <- plot_graph(g, legend = "none")
  
  g2 <- plot_grid(g2c,g2b,g2a, rel_widths = c(1,1,1.4), nrow=1)
  
  g <- plot_grid(g1,g2, nrow = 2)
  
  if(mdp_bool) {
    g_opp_ls <- define_landscape_graph()
    #make it wider
    x = c(3.25, 0.5, 2.25, -1.25,  4.25,  0.5, 2.25, 0.5, 6, 4.25, 6, 2.25, 7.75, 4.25, 6, 3.25)
    y = c(1,      1.75,    1.75, 2.5, 1.75,    2.5,  2.5,  3.25,   1.75,   2.5,2.5,   3.25,   2.5,3.25,   3.25,  4)
    y = rev(y)
    joint_prob_mdp <- joint_prob2 %>% filter(condition == "mdp")
    g <- compute_combo_graph(joint_probs = joint_prob_mdp, min_degree=0, min_prob = 0, state = TRUE)
    g <- g %>% 
      activate("nodes") %>%
      left_join(val_df)
    
    g1 <- ggraph(g_opp_ls, layout='manual', x=x, y=y) + 
      geom_node_point(aes(fill=val), size = 9, shape = 21, color = "black") + 
      geom_edge_link(aes(edge_width = weight),
                     arrow = arrow(ends = "last", length = unit(0.2, "cm"), angle = 30, type = "open"), 
                     show.legend = FALSE,
                     angle_calc = 'along',
                     label_dodge = unit(-8, 'mm'),
                     label_size = 7,
                     end_cap = circle(4, 'mm')) +
      scale_edge_width(range = c(0.2, 4)) + 
      #geom_node_text(aes(label = label), size = 3, color = "white") + 
      scale_fill_gradient2(low = "yellow", mid = "white", high = "black") +
      theme(text = element_text(size = 14), legend.position = "right") + 
      labs(fill = "Fitness",
           tag = "A")
    
    g2 <- ggraph(g, layout = 'manual', x=x, y=y) + 
      geom_node_point(aes(fill=val), size = 9, shape = 21, color = "black") + 
      geom_edge_link(aes(edge_width = weight),
                     arrow = arrow(ends = "last", length = unit(0.2, "cm"), angle = 30, type = "open"), 
                     show.legend = FALSE,
                     angle_calc = 'along',
                     label_dodge = unit(-8, 'mm'),
                     label_size = 7,
                     end_cap = circle(4, 'mm')) +
      scale_edge_width(range = c(0.2, 4)) + 
      #geom_node_text(aes(label = label), size = 3, color = "white") + 
      scale_fill_gradient2(low = "black", mid = "white", high = "yellow") +
      theme(text = element_text(size = 14), legend.position = "right") + 
      labs(fill = "state \n value", 
           tag= "B")
    
    g1 <- plot_grid(g1, g2, ncol = 2)
    g2 <- landscape_summary(out=out)
    
    g <- plot_grid(g1,g2, nrow =2)
    
  }
  
  return(g)
}

mira_traversal_plot2 <- function(out) {
  joint_prob = out[[9]] %>% 
    pivot_wider(id_cols = c(drug1, drug2, reset),names_from = condition, values_from = joint_prob)
  load("data/results/val_mat.Rda")
  
  g1a <- ggplot(joint_prob, aes(x = evodm, y = optimal_policy)) + 
    geom_point() + 
    geom_smooth(method = "lm", se=FALSE) + 
    stat_cor() + 
    theme(text = element_text(size = 16)) + 
    labs(y = "state transition probability \nunder MDP condition", 
         x = "state transition probability \nunder RL-fit condition",
         tag = "A")
  
  g1b <- ggplot(joint_prob, aes(x = evodm_sv, y = optimal_policy)) + 
    geom_point() + 
    geom_smooth(method = "lm", se=FALSE) + 
    stat_cor() + 
    theme(text = element_text(size = 16)) + 
    labs(y = "state transition probability \nunder MDP condition",
         x = "state transition probability \nunder RL-genotype condition",
         tag = "B")
  
  g1c <- ggplot(joint_prob, aes(x = evodm, y = naive)) + 
    geom_point() + 
    geom_smooth(method = "lm", se=FALSE) + 
    stat_cor() + 
    theme(text = element_text(size = 16)) + 
    labs(y = "state transition probability \nunder random condition", 
         x = "state transition probability \nunder RL-fit condition",
         tag = "C")
  
  val_mat <- val_mat[,1:20]
  val_df <- data.frame(name = 1:16, val = scale(rowMeans(val_mat))) %>%
    mutate(decrement = -val) %>% as_tibble()
  df <- out[[10]] %>% group_by(condition, state) %>% 
    summarise(n = n())
  
  g3 <- ggplot(data = df, mapping = aes(x = state, y = n, fill=condition)) + 
    geom_col(position="dodge") + 
    scale_x_continuous(n.breaks = 13, expand = c(0,0)) + 
    theme_classic() + 
    scale_y_continuous(expand = c(0,0), limits = c(-10000, 160000)) + 
    scale_fill_discrete(labels = c("RL_fit", "RL_genotype", "random", "MDP")) +
    theme(text = element_text(size=16)) + 
    labs(tag = "D")
  
  g3b <- ggplot(data = val_df, aes(fill =val)) +
    geom_rect(aes(xmin = as.numeric(name) -0.5, xmax = as.numeric(name) + 0.5, ymin = -6000, ymax = 0, fill = val)) + 
    scale_fill_viridis() + 
    theme_nothing() + theme(legend.position = "right", text = element_text(size = 16),
                            legend.key.height = unit(1, units = "cm")) + 
    labs(fill = "state \n value") 
  leg <- lemon::g_legend(g3b)
  
  leg <- plot_grid(leg, scale = 0.5) + 
    theme(plot.margin = unit(c(-3,-3,-3,-3), "cm"))
  
  g3b <- g3b + theme(legend.position = "none")
  
  g3 <- g3 + 
    annotation_custom(ggplotGrob(g3b), xmin = -0.2, xmax = 17.25, ymin = -10000, ymax = 0)# +
  #annotation_custom(ggplotGrob(leg), xmin = 16.5, xmax = 19, ymin = 0, ymax = 30000)
  
  plot_grid(g1a,g1b,g1c,g3)
}

########Policy Networks figure############
plot_policy_networks <- function(out) {
  joint_prob = out[[6]]
  joint_prob_sv <- joint_prob %>% filter(condition == "evodm_sv")
  g <- compute_combo_graph(joint_probs = joint_prob_sv)
  g2a <- ggraph(g) + 
    geom_node_point(size = 14, fill = "navy", colour = "navy") + 
    geom_edge_fan(aes(edge_width = weight/100), strength = 1.5,
                  arrow = arrow(ends = "last", length = unit(1, "cm"), angle = 15, type = "closed"), 
                  show.legend = FALSE,
                  angle_calc = 'along',
                  label_dodge = unit(-8, 'mm'),
                  label_size = 7,
                  end_cap = circle(10, 'mm')) +
    geom_node_text(aes(label = label), size = 6, nudge_x = -0.3, nudge_y = 0.015) +
    labs(tag = "B: evodm_genotype") + 
    theme(text = element_text(size = 12))
  
  joint_prob_fit <- joint_prob %>% filter(condition == "evodm")
  g <- compute_combo_graph(joint_probs = joint_prob_fit)
  g2b <- ggraph(g) + 
    geom_node_point(size = 14, fill = "navy", colour = "navy") + 
    geom_edge_fan(aes(edge_width = weight/100), strength = 1.5,
                  arrow = arrow(ends = "last", length = unit(1, "cm"), angle = 15, type = "closed"), 
                  show.legend = FALSE,
                  angle_calc = 'along',
                  label_dodge = unit(-8, 'mm'),
                  label_size = 6,
                  end_cap = circle(10, 'mm')) +
    geom_node_text(aes(label = label), size = 7, nudge_x = -0.23, nudge_y = 0.015) +
    labs(tag = "B: evodm") + 
    theme(text = element_text(size = 12))
  
  joint_prob_mdp <- joint_prob %>% filter(condition == "optimal_policy")
  g <- compute_combo_graph(joint_probs = joint_prob_mdp)
  g2c <- ggraph(g) + 
    geom_node_point(size = 14, fill = "navy", colour = "navy") + 
    geom_edge_fan(aes(edge_width = weight/100), strength = 1.5,
                  arrow = arrow(ends = "last", length = unit(1, "cm"), angle = 15, type = "closed"), 
                  show.legend = FALSE,
                  angle_calc = 'along',
                  label_dodge = unit(-8, 'mm'),
                  label_size = 6,
                  end_cap = circle(10, 'mm')) +
    geom_node_text(aes(label = label), size = 7, nudge_x = -0.23, nudge_y = 0.015) +
    labs(tag = "B: mdp") + 
    theme(text = element_text(size = 12))
  
  g2 <- plot_grid(g2a,g2b,g2c)
  
  return(g2)
}

#########Two-drug combos plot######## 
plot_2drug <- function(df_combos_long, df_combos_big) {
  
  #more plots of the combos
  #another plot about the two-drug combos
  g1 <- ggplot(df_combos_long, aes(x = combo, y = average_fitness)) + 
    #geom_boxplot() + geom_quasirandom() + 
    #geom_violin() + 
    geom_boxplot() + 
    geom_rect(aes(ymin = 0, ymax = 0.25, fill = correl, 
                  xmin = as.numeric(combo) -0.5, xmax = as.numeric(combo) +0.5)) + 
    theme(text = element_text(size = 16), 
          axis.text.x = element_blank()) + 
    scale_y_continuous(expand = c(0,0)) + 
    labs(x = "two-drug combination", 
         y = "mean population fitness", 
         fill = "landscape \n correlation") + 
    scale_fill_gradient2(low = "blue", mid = "white", high = "red")
  
  g2 <- ggplot(df_combos_big, aes(x = factor(starting_state), y = average_fitness)) + 
    geom_boxplot() +geom_quasirandom(alpha = 0.4) +
    labs(x = "starting genotype", 
         y = "mean population fitness") + 
    theme(text = element_text(size = 16)) 
  
  g <- plot_grid(g1,g2)
  return(g)
}

#####Policy Plot 2#####
plot_opt_policy <- function(out) {
  df <- out[[5]]
  g <- ggplot(df, aes(x = time_step, y = state, fill = drug)) + 
    geom_tile() + 
    scale_x_continuous(expand = c(0,0)) + 
    scale_y_continuous(expand=c(0,0)) + 
    theme(text=element_text(size=16)) + 
    labs(y="Genotype", 
         x="Time Step", 
         title = "Optimal drug cycling policy")
  
  return(g)
}

################hyperparameter tuning######################

###gamma, batch size, lr, update_target_every all need to be represented here

plot_hp_tune <- function() {
  
  mems_lr <- prep_lr() #%>% 
  #filter(lr > 1e-5)
  
  g1 <- ggplot(mems_lr, aes(x = original_ep, y = fitness, color = lr)) + 
    geom_line() + 
    theme(text = element_text(size = 16)) #+ 
  facet_wrap(~lr)
  
  g2 <-ggplot(mems_lr, aes(x = factor(lr), y = fitness)) + 
    geom_violin()
  
  mems_batchsize <- prep_batch_size()
  
  g3 <- ggplot(mems_batchsize, aes(x = original_ep, y = fitness, color = batch_size)) + 
    geom_line() + 
    theme(text = element_text(size = 16))
  
  g4 <- ggplot(mems_batchsize, aes(x = factor(batch_size), y = fitness)) + 
    geom_violin()
  
  mems_updatetarget <- prep_update_target()
  g5 <- ggplot(mems_batchsize, aes(x = original_ep, y = fitness, color = update_target)) + 
    geom_line() + 
    theme(text = element_text(size = 16))
  g <- plot_grid(g1,g3)
  return(g)
}

#### Procedurally generated landscapes figure#####
#### MDP parameter sweep######
plot_random_mdp <- function(){
  load(file = here("data", "results", "normalized_mdp_sweep_random.Rda"))
  df_prob = out[[1]]
  df_sum = out[[2]]
  df_sum <- df_sum %>% group_by(sigma, num_drugs, N) %>% 
    summarise(benefit = mean(benefit), sd = sd(benefit))
  
  g1 <- ggplot(df_sum, aes(x = sigma, y = num_drugs, fill = benefit)) + 
    geom_tile() + 
    facet_wrap(~N, labeller = labeller(.cols = label_both)) + 
    scale_fill_viridis_c() + 
    labs(x = "sigma (epistasis coefficient)",
         y = "number of available drugs", 
         fill = "fitness benefit (random - mdp)") + 
    scale_x_continuous(expand = c(0,0)) + 
    scale_y_continuous(expand = c(0,0)) + 
    theme(legend.position = "top", 
          legend.key.width = unit(2.5, "cm"),
          text = element_text(size = 26)) + 
    guides(fill = guide_colorbar(title.position = "top",
                                 title.hjust = unit(-0, "cm")))
  
  g2 <- ggplot(df_sum, aes(x = factor(N), y = benefit)) +
    geom_violin()
  
  df_sum %>% group_by(N) %>% 
    summarise(benefit = mean(benefit))
  
  df_sum %>% group_by(num_drugs) %>%
    summarise(benefit = mean(benefit)) 
  
  df_sum %>% group_by(sigma) %>%
    summarise(benefit = mean(benefit)) 
  
  return(g1)
  
}


####MDP Mira stuff####
plot_mira_mdp <- function(){
  load(here("data", "results", "not_normalized_mdp_sweep.Rda"))
  df_prob <-out[[1]]
  df_sum <- out[[2]]
  
  g1 <- ggplot(df_sum, aes(x = gamma, y = fitness)) + geom_line() + 
    labs(y = "Average Fitness") + 
    scale_x_continuous(expand = c(0,0)) + 
    scale_y_continuous(expand = c(0,0)) + 
    theme(text = element_text(size= 16)) 
  
  g2 <- ggplot(df_prob, aes(x = gamma, y = prob_selection, fill = factor(drug))) + 
    geom_area() + 
    labs(y = "probability of selection", fill = "drug") + 
    scale_x_continuous(expand = c(0,0)) + 
    scale_y_continuous(expand = c(0,0)) + 
    theme(text = element_text(size= 16),
          legend.position = "top") 
  return(plot_grid(g1, g2))
}

###Actually plot everything

#Performance Plot
#base case
g <- mira_perf_plot(out = out_hm, out_noise = out_noise, df_combos_long = df_combos_long, 
                    df_combos = df_combos, reset = 20, replicates = 30)
png(filename = here("figs", "mira_landscapes20.png"), 
    width = 1000, height = 600)
g
dev.off()

g <- mira_perf_supps(out=out_hm)
png(filename = here("figs", "mira_landscapes20_supps.png"), 
    width = 1000, height = 600)
g
dev.off()
#quadmut starting position
# g <- mira_perf_plot(out = out_hm, df_combos_long = df_combos_long, 
#                     df_combos = df_combos, reset = 20, replicates = 100, 
#                     quad_mut = TRUE)
# png(filename = here("figs", "mira_landscapes20_quadmut.png"), 
#     width = 1000, height = 600)
# g
# dev.off()


# #optimize over total resistance instead of single drug resistance
# g <- mira_perf_plot(out = out_hm, df_combos_long = df_combos_long, 
#                     df_combos = df_combos, reset = 20, replicates = 100, 
#                     quad_mut = FALSE, total_res = TRUE)
# png(filename = here("figs", "mira_landscapes20_totalres.png"), 
#     width = 1000, height = 600)
# g
# dev.off()
# #policy plot


g <- mira_policy_plot(out = out_hm)
png(filename = here("figs", "mira_policy.png"), 
    width = 1000, height = 600)
g
dev.off()

#####PCA policy cluster stuff
out <- mira_policy_plot(out = out_hm, cluster_bool = TRUE)
png(filename= here("figs", "mira_policy_pca.png"),
    width = 1000, height = 600)
out[[1]]
dev.off()

png(filename= here("figs", "mira_policy_pca2.png"),
    width = 1000, height = 600)
out[[2]]
dev.off()


#Plot traversal figure
g <- mira_traversal_plot(out = out_hm)
ggsave(here("figs", "mira_traversal.tiff"),plot=g, device = "tiff", units="in", width=7, height=5, dpi = 700)
ggsave(here("figs", "mira_traversal.png"), plot=g,device = "png", units="in", width=7, height=5, dpi = 700)

g <- mira_traversal_plot(out=out_hm, mdp_bool = TRUE)
ggsave(here("figs", "mira_traversal_mdp.tiff"), plot=g,device = "tiff", units="in", width=10, height=8, dpi = 500)
ggsave(here("figs", "mira_traversal_mdp.png"),plot=g, device = "png", units="in", width=10, height=8, dpi = 500)


#optimal policy supps fig
g <- plot_opt_policy(out=out_hm)
ggsave(here("figs", "opt_policy.tiff"), plot=g, device = "tiff", units="in", width=5, height=4, dpi = 300)
ggsave(here("figs", "opt_policy.png"), plot=g,device = "png", units="in", width=5, height=4, dpi = 300)
#tiff(filename = here("figs", "mira_traversal.tiff"), 
#    width = 1000, height = 600, res = 1000)
#g
#dev.off()

g <- mira_traversal_plot2(out = out_hm)
png(filename = here("figs", "mira_traversal2.png"), 
    width = 1000, height = 600)
g
dev.off()


#random landscapes figure

##Supps##

#Sensitivity analyses around training on mira landscapes
g <- mira_perf_sens(out=out_hm)
png(filename = here("figs", "mira_perf_sens.png"), 
    width = 1000, height = 600)
g
dev.off()


#two-drug sweep
g <- plot_2drug(df_combos_long = df_combos_long, df_combos_big = df_combos_big)
png(filename = here("figs", "two_drug_sweep.png"), 
    width = 1000, height = 600)
g
dev.off()

#MDP Random Sweep 4
png(filename = here("figs", "random_mdp_sweep.png"), width = 800, height = 800)
plot_random_mdp()
dev.off()

#MDP Mira Sweep
png(filename = here("figs", "mira_mdp_sweep.png"), 
    width = 800, height = 400)
plot_mira_mdp()
dev.off()

#hyper-parameter tuning
png(filename = here("figs", "hp_tune_plot.png"), 
    width = 1000, height = 600)
plot_hp_tune()
dev.off()
