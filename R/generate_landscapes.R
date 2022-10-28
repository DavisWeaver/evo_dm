#File contains functions to generate random landscapes of varying num_drugs, N, and sigma
library(reticulate)
library(here)
source(here("R", "clean_evol_out.R"))
source_python(here("load.py"))
set.seed(12564)
N_range = c(4, 10)
num_drugs_range = c(3, 15)
sigma_range = c(0,5)

for(n in N_range) {
  for(d in num_drugs_range) {
    for(s in sigma_range) {
      N_vec <- replicate(n = 3, expr = n)
      drug_list <- 
        lapply(N_vec, FUN = define_drugs_r, num_drugs = d, sigma = s, CS = FALSE)
      save(drug_list, 
           file = here(
             "data", 
             paste0(
               "random_landscapes_N",n, "D", d, "S", s, ".Rda"
               )
             )
           )
    }
  }
}


