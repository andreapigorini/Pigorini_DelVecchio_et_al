library(ggplot2)
library(dplyr)
library(tidyr)
library(lmerTest)

analyze_band <- function(file_path, band_name) {
  
  dat <- read.csv(file_path)
  
  # differenze
  dat <- dat %>%
    mutate(
      dif_conn_e = conn_post_early - conn_bl,
      dif_conn_l = conn_post_late - conn_bl,
      clust = as.character(clust)
    ) %>%
    filter(clust_lobe %in% c("GUF","LUF","GUP","LUP")) %>%
    mutate(
      clust = factor(clust, levels = c("GU","LU")),
      region = case_when(
        clust_lobe %in% c("GUF","LUF") ~ "frontal",
        clust_lobe %in% c("GUP","LUP") ~ "parietal"
      )
    )
  
  # clust, task, region combinations
  clust_task_combos <- expand.grid(
    Clust = unique(dat$clust),
    Task = unique(dat$task),
    Region = unique(dat$region),
    stringsAsFactors = FALSE
  )
  
  results <- list()
  k <- 1
  
  for (cond in c("early","late")) {
    for (i in 1:nrow(clust_task_combos)) {
      clust_value <- clust_task_combos$Clust[i]
      task_value  <- clust_task_combos$Task[i]
      region_value <- clust_task_combos$Region[i]
      
      # subset
      subset <- dat %>%
        filter(clust == clust_value, task == task_value, region == region_value, dist > 30)
      
      if(nrow(subset) > 1) {
        # model
        response_var <- ifelse(cond=="early","dif_conn_e","dif_conn_l")
        formula_str <- as.formula(paste0(response_var," ~ 1 + (1|subject)"))
        model <- lmer(formula_str, data = subset)
        
        beta <- summary(model)$coefficients[, "Estimate"]
        std <- summary(model)$coefficients[, "Std. Error"]
        p_val    <- summary(model)$coefficients[, "Pr(>|t|)"]
        
        results[[k]] <- data.frame(
          clust = clust_value,
          band = band_name,
          condition = cond,
          region = region_value,
          task = task_value,
          beta = beta,
          std = std,
          p_val_starting = p_val
        )
        k <- k+1
      }
    }
  }
  
  do.call(rbind, results)
}

# --- beta and gamma bands analysis ---
#PATHS TO BE CHANGED!!
fname_beta <- "~/data/mbase/results/tabs/table_connectivity_all_lobes_beta.csv"
fname_gamma <- "~/data/mbase/results/tabs/table_connectivity_all_lobes_gamma.csv"

results_beta  <- analyze_band(fname_beta, "beta")
results_gamma <- analyze_band(fname_gamma, "gamma")

# --- union and correction pval ---
results_df <- bind_rows(results_beta, results_gamma) %>%
  mutate(
    p_val_corr = p.adjust(p_val_starting, method = "bonferroni"),
    significant = ifelse(p_val_corr < 0.05, 1, 0),
    direction = case_when(
      significant == 1 & beta > 0 ~ "Above Zero",
      significant == 1 & beta < 0 ~ "Below Zero",
      TRUE ~ "Not Significant"
    )
  )

