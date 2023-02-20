require(tidyr)
require(readr)
require(pheatmap)
require(dplyr)
require(svglite)
require(viridis)
require(ggplot2)
require(RColorBrewer)
heatmap_colors <- colorRampPalette(brewer.pal(n = 7, name ="Reds"))(100)
heatmap_colors2 <- colorRampPalette(colors = c("blue", "white", "red"))(100)
heatmap_colors3 <- cividis(100)
df <- read.csv("output/timecourse_heatmaps/timecourse_bestfit_cells.csv")
df <- df %>% select(time, species, value, Infection)

df_wide <- df %>% pivot_wider(names_from = species, values_from = value)

# only focus on some subsets?

df_wide <- df_wide %>% select(time, Infection, Naive, Precursor, Th1, Th1_c, Th1_mem, CD4_All)
colnames(df_wide) <- c("time", "Infection", "Naive", "Prec.", "Eff.", "Chr.", "Mem.", "CD4 All")

prep_heatmap <- function(df, Infection){
  df <- df[df$Infection == Infection,]
  df$Infection <- NULL
  rownames(df) <- df$time
  df$time <- NULL
  df <- t(df)
  df <- log10(df)
  df[df<0] <- 0
  df[is.na(df)] <- 0
  return(df)
}

df_arm <- prep_heatmap(df_wide, "Arm")
df_cl13 <- prep_heatmap(df_wide, "Cl13")


breaks <- seq(1,6, length.out = 101)
cellwidth = 0.1
cellheight = 4.5

require(svglite)
require(dplyr)
require(grid)
# utility functions used for plotting

save_pheatmap <- function(x, filename, width=8, height=7) {
  stopifnot(!missing(x))
  stopifnot(!missing(filename))
  svg(filename, width=width, height=height)
  grid::grid.newpage()
  grid::grid.draw(x$gtable)
  dev.off()
}

w <- 2.6
h <- 1.

proc_df <- function(df){
  df 
}

p1 <- pheatmap(df_arm, cluster_rows = F, cluster_cols = F, breaks = breaks, color = heatmap_colors3,
         cellwidth = cellwidth,
         cellheight = cellheight,
         filename = "figures/timecourses/timecourse_bestfit_heatmap_arm.png", dpi = 300,
         width = w,
         height = h,
         fontsize = 6)

save_pheatmap(p1, "figures/timecourses/timecourse_bestfit_heatmap_arm.svg", width = w, height = h)

p2 <- pheatmap(df_cl13, cluster_rows = F, cluster_cols = F, breaks = breaks, color = heatmap_colors3,
         cellwidth = cellwidth,
         cellheight = cellheight,
         filename = "figures/timecourses/timecourse_bestfit_heatmap_cl13.png", dpi = 300,
         width = w,
         height = h,
         fontsize = 6)
save_pheatmap(p2, "figures/timecourses/timecourse_bestfit_heatmap_cl13.svg", width = w, height = h)

