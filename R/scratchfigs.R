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
theme_set(theme_classic())


df <- prep_results()


ggplot(df, aes(x = fitness, fill = condition)) + geom_density(alpha = 0.3)


