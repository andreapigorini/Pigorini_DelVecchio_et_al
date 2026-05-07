library(R.matlab)
library(ggplot2)
library(lmerTest)
library(dplyr)
library(tidyr)

# Load CCEP connectivity results.
results = read.csv('~/data/Pigorini_DelVecchio_et_al/results/ccep_results.csv')

# Define binary effective connectivity as the presence of at least one significant response.
results$conn <- results$nresp > 0

# Encode subject as a categorical variable for mixed-effects modeling.
results$subj <- as.factor(results$subj)

# Set response-type levels, with gamma-responsive contacts used as the reference category.
results$respt <- factor(results$respt, levels=c('gamma', 'lfp', 'noresp')) # levels=c('gamma', 'lfp', 'noresp')

# Remove rows with missing connectivity values.
results <- results %>% drop_na(conn)

# Inspect the resulting dataframe.
summary(results)

# Fit a binomial generalized linear mixed-effects model testing whether effective
# connectivity differs across response types, with subject as a random intercept.
m <- glmer(conn ~ respt + (1|subj), data = results, family = binomial)

# Print model summary without the fixed-effect correlation matrix.
summary(m, corr = FALSE)

# Estimate marginal predicted probabilities of effective connectivity for each response type.
predict(m, type = "response", re.form = NA, newdata = data.frame(respt = c("gamma", "lfp", "noresp")))

# Convert fixed-effect log-odds coefficients to odds ratios.
exp(fixef(m))

# Plot subject-level connectivity estimates grouped by response type.
ggplot(results, aes(subj, as.numeric(conn), color=respt)) + stat_summary(fun.data = "mean_cl_boot") + coord_flip()

# Compute the overall proportion of connected contacts for each response type.
results %>% group_by(respt) %>% summarise(prop=mean(conn))

# Compute subject-level connectivity proportions for each response type.
results_subj <- results %>% group_by(subj, respt) %>% summarise(prop=mean(conn))

# Plot subject-level connectivity proportions across response types.
ggplot(results_subj, aes(respt, prop, fill=respt)) + geom_boxplot() + geom_jitter(width=0.05) + scale_fill_manual(values = c("gamma" = "orange",  "lfp" = "purple", "noresp" = "grey")) + theme_minimal()


