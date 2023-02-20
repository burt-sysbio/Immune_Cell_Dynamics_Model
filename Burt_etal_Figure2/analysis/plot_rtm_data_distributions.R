# plot distribution of rtm-normalized data

require(readr)
require(dplyr)
require(tidyr)
require(ggplot2)
require(stringr)
source("src/paper_theme_R.R")

mypath <- "data/data_rtm/"
readnames <- list.files(path = mypath, pattern = "_rtm_", full.names = T)

mynames <- c("Craw. 1", "Craw. 2", "Craw. 3", "Craw. 4",
             "Nir 1", "Nir 2", "Peine 1", "Peine 2", "Peine 3", "Peine 4",
             "Ilott 1", "Pros. 1")

readfun <- function(readname, studyname){
  df <- read_csv(readname)
  df$study <- studyname
  return(df)
}

distplot <- function(df, ycol, sname, ymin = -10, ymax = 10, breaks = c(-2,-1,0,1,2), width =4, height = 2){
  sname <- paste0("figures/data_distributions/data_processed_", sname, ".png")
  
  df <- df %>% filter(name == ycol)
  g <- ggplot(data = df, aes(x = study, y = value)) +
    geom_boxplot(outlier.shape = NA) +
    theme_R() +
    scale_y_continuous(limits = c(ymin, ymax), breaks = breaks) +
    theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1)) +
    labs(y = "expr. norm.", x = "")
  print(g)
  ggsave(sname, bg = "white", width = width, height = height, dpi = 300)
}


df_list <- mapply(readfun, readnames, mynames, SIMPLIFY = F)
df <- bind_rows(df_list)
df <- df %>% select(study, value, val_norm, val_norm_rtm2) %>% pivot_longer(-study)

#distplot(df, "value", "norm_log22")
#distplot(df, "val_norm", "norm_d0")
distplot(df, "val_norm_rtm2", "norm_rtm", ymin = -2, ymax = 2.5, width = 2, height = 1.9)


g <- ggplot(data = df, aes(x = study, y = value, colour = name)) +
  geom_boxplot(outlier.shape = NA) +
  theme_bw() +
  theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1)) +
  labs(y = "expr. norm.", x = "")
print(g)
ggsave("figures/data_distributions/data_processed_all_norms.png", bg = "white", width =  5, height = 3)
