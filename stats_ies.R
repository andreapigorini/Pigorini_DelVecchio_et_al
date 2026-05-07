library(ggplot2)
library(lmerTest)
library(tidyr)
library(dplyr)
library(car)
library(emmeans)
library(lme4)

# Load the contact-level table containing stimulation responses, contact classes,
# and cortical gradient values.
fname_df <- "~/data/Pigorini_DelVecchio_et_al/results/trains_x_stat.csv"

data <- read.csv(fname_df)
summary(data)

# Reshape the dataset from wide to long format, obtaining one row per
# subject-contact-modality combination.
long <- data %>%
  pivot_longer(
    cols = matches("^(video|somato|audio)_(resp|gamma|lfp|none|anat)$"),
    names_to = c("modality", ".value"),
    names_pattern = "^(video|somato|audio)_(resp|gamma|lfp|none|anat)$"
  ) %>%
  mutate(
    # Define contact type according to gamma, LFP-only, or unresponsive classification.
    ch_type = case_when(
      gamma == 1 ~ "gamma",
      lfp   == 1 ~ "lfp",
      none  == 1 ~ "none",
      TRUE ~ NA_character_
    )
  ) %>%
  select(subj, ch_name, modality, resp, ch_type, mapval)

# Remove incomplete rows and encode grouping variables as factors.
long <- long %>%
  filter(!is.na(ch_type), !is.na(resp)) %>%
  mutate(
    ch_type = factor(ch_type),
    subj = factor(subj)
  )

# Create a unique contact identifier nested within subject.
long <- long %>%
  mutate(ch_id = interaction(subj, ch_name, drop = TRUE))

# Ensure correct factor encoding and center the cortical gradient variable.
long$subj <- factor(long$subj)
long$ch_type <- factor(long$ch_type)
long$mapval_c <- as.numeric(scale(long$mapval, center = TRUE, scale = FALSE))

# Fit a baseline binomial mixed-effects model testing whether elicitation probability
# differs across contact types, with subject as a random intercept.
m1 <- glmer(resp ~ ch_type + (1|subj), data=long, family="binomial")
summary(m1)

# Estimate marginal predicted elicitation probabilities for each contact type.
nd <- data.frame(ch_type = levels(long$ch_type))
nd$prob <- plogis(predict(m1, newdata = nd, re.form = NA))
nd


# Additive model: test independent effects of contact type and cortical gradient.
m_add <- glmer(resp ~ ch_type + mapval_c + (1|subj) + (1|modality), data = long, family = binomial)
summary(m_add)

# Interaction model: test whether contact-type effects vary along the cortical gradient.
m_int <- glmer(resp ~ ch_type * mapval_c + (1|subj) + (1|ch_id) +(1|modality), data = long, family = binomial)
summary(m_int)

# Type-II tests for fixed effects in the additive and interaction models.
car::Anova(m_add, type = 2)
car::Anova(m_int, type = 2)

# Select representative gradient values corresponding to low, intermediate,
# and high positions along the centered gradient.
vals <- as.numeric(quantile(long$mapval_c, c(.1, .5, .9), na.rm = TRUE))

# Estimate contact-type marginal means from the additive model at selected gradient values.
emm_grid <- emmeans(
  m_add,
  ~ ch_type | mapval_c,
  at = list(mapval_c = vals),
  type = "response")
emm_grid

# Pairwise post-hoc comparisons among contact types at each gradient value.
pairs(emm_grid, by = "mapval_c", adjust = "tukey")

# Estimate contact-type marginal means from the interaction model at selected gradient values.
emm_grid <- emmeans(
  m_int,
  ~ ch_type | mapval_c,
  at = list(mapval_c = vals),
  type = "response")
emm_grid

# Pairwise post-hoc comparisons from the interaction model.
pairs(emm_grid, by = "mapval_c", adjust = "tukey")

# Generate a grid of gradient values for plotting predicted elicitation probabilities.
vals <- seq(min(long$mapval_c, na.rm = TRUE),
            max(long$mapval_c, na.rm = TRUE),
            length.out = 10)

# Plot model-predicted elicitation probabilities across the cortical gradient.
p <- emmip(m_int, ch_type ~ mapval_c,
           at = list(mapval_c = vals), dodge = 0,
           CIs = TRUE, type = "response", style = "numeric") +
  coord_cartesian(ylim = c(0, 1)) +
  theme_minimal()

p + scale_color_manual(
  values = c(
    "gamma" = "orange",
    "lfp"   = "purple"
  ),
  na.value = "grey60"
)

# Plot observed response rates by modality and contact type.
pd <- position_dodge(width = 0.8)

ggplot(long, aes(x = modality, y = resp, fill = ch_type, group = ch_type)) +
  stat_summary(fun = mean, geom = "bar", position = pd, width = 0.7) +
  stat_summary(fun.data = mean_cl_boot, geom = "errorbar",
               position = pd, width = 0.2) +
  scale_fill_manual(values = c(gamma = "orange", lfp = "purple", uc = "grey70"),
                    breaks = c("gamma","lfp","uc")) +
  scale_y_continuous(limits = c(0, 1)) +
  labs(y = "P(resp = 1)", x = NULL, fill = NULL) +
  theme_classic()

# Compute observed proportions and binomial confidence intervals by modality and contact type.
obs_sum <- long %>%
  group_by(modality, ch_type) %>%
  summarise(
    n = n(),
    n_resp = sum(resp == 1, na.rm = TRUE),
    prop_resp = n_resp / n,
    .groups = "drop"
  ) %>%
  rowwise() %>%
  mutate(
    ci = list(prop.test(n_resp, n)$conf.int),
    ci_low = ci[[1]],
    ci_high = ci[[2]]
  ) %>%
  ungroup() %>%
  select(-ci)
obs_sum


# Fit a binomial mixed-effects model testing contact-type, modality,
# and contact-type-by-modality effects while controlling for gradient position.
m_cm <- glmer(
  resp ~ ch_type * modality + mapval_c + (1 | subj) + (1 | ch_id),
  data = long,
  family = binomial
)

summary(m_cm)

# Type-III test of fixed effects for the contact-type-by-modality model.
car::Anova(m_cm, type = 3)

# Use sum-to-zero contrasts for Type-III tests.
options(contrasts = c("contr.sum", "contr.poly"))
car::Anova(m_cm, type = 3)

# Estimate marginal probabilities for each contact type within each modality.
emm_ct_by_mod <- emmeans(m_cm, ~ ch_type | modality, type = "response")
emm_ct_by_mod

# Pairwise post-hoc comparisons among contact types within each modality.
pairs(emm_ct_by_mod, adjust = "tukey")