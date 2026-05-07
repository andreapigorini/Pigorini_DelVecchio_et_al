library(R.matlab)
library(ggplot2)
library(lmerTest)
library(dplyr)
library(tidyr)
results = read.csv('~/data/Pigorini_DelVecchio_et_al/results/ccep_results.csv')
results$conn <- results$nresp > 0
results$subj <- as.factor(results$subj)
results$respt <- factor(results$respt, levels=c('gamma', 'lfp', 'noresp')) # levels=c('gamma', 'lfp', 'noresp')
results <- results %>% drop_na(conn)
summary(results)
m <- glmer(conn ~ respt + (1|subj), data = results, family = binomial)
summary(m, corr = FALSE)
predict(m, type = "response", re.form = NA, newdata = data.frame(respt = c("gamma", "lfp", "noresp")))
exp(fixef(m))
ggplot(results, aes(subj, as.numeric(conn), color=respt)) + stat_summary(fun.data = "mean_cl_boot") + coord_flip()
results %>% group_by(respt) %>% summarise(prop=mean(conn))
results_subj <- results %>% group_by(subj, respt) %>% summarise(prop=mean(conn))
ggplot(results_subj, aes(respt, prop, fill=respt)) + geom_boxplot() + geom_jitter(width=0.05) + scale_fill_manual(values = c("gamma" = "orange",  "lfp" = "purple", "noresp" = "grey")) + theme_minimal()



