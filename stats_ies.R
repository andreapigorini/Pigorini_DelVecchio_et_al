library(ggplot2)
library(lmerTest)
library(tidyr)
library(dplyr)
library(car)
library(emmeans)
library(lme4)

fname_df <- "~/data/mbase/tabs/df_trains_x_stat.csv"

data <- read.csv(fname_df)
summary(data)

long <- data %>%
  pivot_longer(
    cols = matches("^(video|somato|audio)_(resp|gamma|lfp|none|anat)$"),
    names_to = c("modality", ".value"),
    names_pattern = "^(video|somato|audio)_(resp|gamma|lfp|none|anat)$"
  ) %>%
  mutate(
    # ch_type from whichever of gamma/lfp/none is 1
    ch_type = case_when(
      gamma == 1 ~ "gamma",
      lfp   == 1 ~ "lfp",
      none  == 1 ~ "none",
      TRUE ~ NA_character_
    )
  ) %>%
  select(subj, ch_name, modality, resp, ch_type, mapval)

long <- long %>%
  filter(!is.na(ch_type), !is.na(resp)) %>%
  mutate(
    ch_type = factor(ch_type),
    subj = factor(subj)
  )

long <- long %>%
  mutate(ch_id = interaction(subj, ch_name, drop = TRUE))

# Ensure factors
long$subj <- factor(long$subj)
long$ch_type <- factor(long$ch_type)
long$mapval_c <- as.numeric(scale(long$mapval, center = TRUE, scale = FALSE))

m1 <- glmer(resp ~ ch_type + (1|subj), data=long, family="binomial")
summary(m1)

nd <- data.frame(ch_type = levels(long$ch_type))
nd$prob <- plogis(predict(m1, newdata = nd, re.form = NA))
nd


# Additive model: same ch_type differences regardless of mapval
m_add <- glmer(resp ~ ch_type + mapval_c + (1|subj) + (1|modality), data = long, family = binomial)
summary(m_add)

# Interaction model: ch_type differences may vary with mapval
m_int <- glmer(resp ~ ch_type * mapval_c + (1|subj) + (1|ch_id) +(1|modality), data = long, family = binomial)
#m_int <- glmer(resp ~ ch_type * mapval_c + (1|subj/ch_name) +(1|modality), data = long, family = binomial)
summary(m_int)

car::Anova(m_add, type = 2)
car::Anova(m_int, type = 2)

vals <- as.numeric(quantile(long$mapval_c, c(.1, .5, .9), na.rm = TRUE))

emm_grid <- emmeans(
  m_add,
  ~ ch_type | mapval_c,
  at = list(mapval_c = vals),
  type = "response")
emm_grid

pairs(emm_grid, by = "mapval_c", adjust = "tukey")

emm_grid <- emmeans(
  m_int,
  ~ ch_type | mapval_c,
  at = list(mapval_c = vals),
  type = "response")
emm_grid

pairs(emm_grid, by = "mapval_c", adjust = "tukey")

vals <- seq(min(long$mapval_c, na.rm = TRUE),
            max(long$mapval_c, na.rm = TRUE),
            length.out = 10)




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

ggplot(long, aes(mapval_c, fill=ch_type)) + geom_histogram(position='stack', bins=5) 
ggplot(long, aes(mapval_c, color=ch_type)) + geom_density() 


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


m_cm <- glmer(
  resp ~ ch_type * modality + mapval_c + (1 | subj) + (1 | ch_id),
  data = long,
  family = binomial
)

summary(m_cm)
car::Anova(m_cm, type = 3)

options(contrasts = c("contr.sum", "contr.poly"))
car::Anova(m_cm, type = 3)

emm_ct_by_mod <- emmeans(m_cm, ~ ch_type | modality, type = "response")
emm_ct_by_mod
pairs(emm_ct_by_mod, adjust = "tukey")





