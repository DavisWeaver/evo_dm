reticulate::use_python("C:/Program Files/Python37/python.exe", required = TRUE) #python path may vary
reticulate::source_python("load.py")

library(ggplot2)
library(dplyr)
library(stringr)
library(here)
#Update this when running

plot_guide <- function(day1, day, platepath, agentpath, savepath, experimental_drug) {
  drug_index <- data.frame(action = -1:15, 
                           drug = c("No_cells","No_drug", "1-AMP", "2-AM", "3-CEC", "4-CTX", "5-ZOX", "6-CXM", 
                                    "7-CRO", "8-AMC", "9-CAZ", "10-CTT", "11-SAM", "12-CPR",
                                    "13-CPD", "14-TZP", "15-FEP"))
  if(!day1) {
    load(file = here("validation", "data", paste0("prev_action", day-1, ".Rda"))) #load prev_action into memory
  } else {
    plate = ''
  }
  plate= format_plate(day1=day1, platepath = platepath, agentpath = agentpath, savefolder = savepath, prev_action = plate, experimental_drug = experimental_drug)
  save(plate, file = here("validation", "data", paste0("prev_action", day, ".Rda"))) #save current actions- this will become prev_action tomorrow
  
  plate = unlist(plate)
  position1 <- str_extract(names(plate), pattern = "[:upper:]")
  position2 <- as.numeric(str_extract(names(plate), pattern = "\\d+"))
  
  df <- data.frame(y = position1, x= position2, action = plate) %>% 
    left_join(drug_index, by = "action")
  
  g <- ggplot(df, aes(x=x, y=y, fill = drug)) + 
    geom_tile(color ='black') + scale_fill_discrete() + 
    geom_text(aes(label = action), color = 'white', size = 8) + 
    scale_x_continuous(breaks = 1:12, limits = c(0,13), expand = c(0,0), position = "top") + 
    scale_y_discrete(limits = rev) 
}

day = 10

day1 = TRUE

experimental_drug = as.integer(12)

g <- plot_guide(day1=day1, platepath = '', 
                agentpath = here('data', 'binaries', 'mira1N4D15_reset20_fit50.p'), 
                day = day, savepath = here('validation', 'data'), 
                experimental_drug = experimental_drug)

pdf(file = here("validation", "guides", paste0("day", day, ".pdf")), 
    width = 6.4, height  = 3.55, paper = "letter")
g
dev.off()
