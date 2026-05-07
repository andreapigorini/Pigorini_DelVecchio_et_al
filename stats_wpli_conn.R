library(ggplot2)
library(dplyr)
library(tidyr)
library(lmerTest)

analyze_band <- function(file_path, band_name) {
  
  # Load connectivity table for the selected frequency band.
  dat <- read.csv(file_path)
  
  # Compute post-stimulus connectivity changes relative to baseline and define
  # anatomical region labels from cluster-lobe codes.
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
  
  # Define all cluster-task-region combinations to be tested independently.
  clust_task_combos <- expand.grid(
    Clust = unique(dat$clust),
    Task = unique(dat$task),
    Region = unique(dat$region),
    stringsAsFactors = FALSE
  )
  
  # Initialize output containers.
  results <- list()
  k <- 1
  
  # Fit one intercept-only mixed model for each time window and cluster-task-region subset.
  for (cond in c("early","late")) {
    for (i in 1:nrow(clust_task_combos)) {
      clust_value <- clust_task_combos$Clust[i]
      task_value  <- clust_task_combos$Task[i]
      region_value <- clust_task_combos$Region[i]
      
      # Select long-range pairs only and restrict to the current cluster-task-region subset.
      subset <- dat %>%
        filter(clust == clust_value, task == task_value, region == region_value, dist > 30)
      
      if(nrow(subset) > 1) {
        
        # Select early or late post-stimulus change relative to baseline.
        response_var <- ifelse(cond=="early","dif_conn_e","dif_conn_l")
        
        # Test whether the mean connectivity change differs from zero,
        # with subject modeled as a random intercept.
        formula_str <- as.formula(paste0(response_var," ~ 1 + (1|subject)"))
        model <- lmer(formula_str, data = subset)
        
        # Extract fixed-effect estimate, standard error, and p-value.
        beta <- summary(model)$coefficients[, "Estimate"]
        std <- summary(model)$coefficients[, "Std. Error"]
        p_val    <- summary(model)$coefficients[, "Pr(>|t|)"]
        
        # Store model output for the current subset.
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
  
  # Return all subset-level model results for the selected frequency band.
  do.call(rbind, results)
}

# --- beta and gamma bands analysis ---
# PATHS TO BE CHANGED!!
fname_beta <- "~/data/Pigorini_DelVecchio_et_al/results/table_connectivity_all_lobes_beta.csv"
fname_gamma <- "~/data/Pigorini_DelVecchio_et_al/results/table_connectivity_all_lobes_gamma.csv"

# Run the same mixed-effects analysis separately for beta and gamma bands.
results_beta  <- analyze_band(fname_beta, "beta")
results_gamma <- analyze_band(fname_gamma, "gamma")

# Combine beta and gamma results, correct p-values across all tests,
# and classify significant effects according to their direction.
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

print(results_df)

write.csv(results_df, "~/data/Pigorini_DelVecchio_et_al/results/Tab_S4.csv", row.names = FALSE)
