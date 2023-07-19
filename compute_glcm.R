#Always load the python stuff in a clean session\
#
reticulate::use_python("C:/Program Files/Python37/python.exe", required = TRUE) #python path may vary

reticulate::source_python("load.py")
library(gtexture)
library(fitscape)
library(dplyr)
library(tidyr)
library(ggplot2)
library(ggrepel)
library(ggbeeswarm)
compute_landscape <- function(N, vals) {
  values <- array(vals, dim = rep(2, N))
  landscape  <- FitLandDF(values)
  return(landscape)
}

main <- function() {
  #metric has to be a function from the coOccuR
  drugs = define_mira_landscapes(as_dict = TRUE)
  
  out <- list()
  for (i in 1:length(drugs)) {
    drug_i <- drugs[[i]]
    N = sqrt(length(drug_i))
    landscape = compute_landscape(N=N, vals = drug_i)
    comat = get_comatrix(landscape, discrete = equal_discrete(3))
    metrics <- compute_all_metrics(comat)
    metrics$id <- as.character(i)
    out[[i]] <- metrics
  }
  
  df <- bind_rows(out)
  drug_index <- data.frame(id = as.character(1:15), 
                           drug = c("AMP", "AM", "CEC", "CTX", "ZOX", "CXM", 
                                     "CRO", "AMC", "CAZ", "CTT", "SAM", "CPR",
                                     "CPD", "TZP", "FEP"),
                           drug_long = c("Ampicillin", "Amoxicillin", "Cefaclor", "Cefotaxime",
                                          "Ceftizoxime", "Cefuroxime", "Ceftriaxone",
                                          "Amoxicillin + Clavulanic acid", "Ceftazidime",
                                          "Cefotetan", "Ampicillin + Sulbactam", 
                                          "Cefprozil", "Cefpodoxime", 
                                          "Pipercillin + Tazobactam", "Cefepime")
  )
  df <- left_join(df, drug_index, by = 'id') %>% 
    mutate_if(is.numeric, scale)
  df <- pivot_longer(df, cols = -c(id, drug, drug_long), values_to = "value", names_to = "feature")
  return(df)
  
}
df <- main()

special_df <- df %>% filter(drug %in% c("CTX", "CPR", "AMP", "SAM","TZP"))
notspecial_df <- df %>% filter(!(drug %in% c("CTX", "CPR", "AMP", "SAM","TZP")))

png(filename = here("figs", "texture_drug.png"),
    width = 1000, height = 600)
ggplot(df, aes(x = feature, y = value)) + geom_boxplot() + 
  geom_quasirandom(data = notspecial_df) + 
  geom_quasirandom(data = special_df, aes(color = drug)) + 
  labs(x = "haralick texture feature", y = "Z-score normalized value") + 
  theme_classic() + 
  theme(text = element_text(size=16), 
        axis.text.x = element_text(angle = 45, vjust = 0.59))
dev.off()


