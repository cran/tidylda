## ----setup, include=FALSE-----------------------------------------------------
knitr::opts_chunk$set(echo = TRUE)

## ----eval = FALSE-------------------------------------------------------------
#  install.packages("remotes")
#  
#  remotes::install_github("tommyjones/tidylda")

## ----example------------------------------------------------------------------
library(tidytext)
library(dplyr)
library(ggplot2)
library(tidyr)
library(tidylda)
library(Matrix)

### Initial set up ---
# load some documents
docs <- nih_sample 

# tokenize using tidytext's unnest_tokens
tidy_docs <- docs |> 
  select(APPLICATION_ID, ABSTRACT_TEXT) |> 
  unnest_tokens(output = word, 
                input = ABSTRACT_TEXT,
                stopwords = stop_words$word,
                token = "ngrams",
                n_min = 1, n = 2) |> 
  count(APPLICATION_ID, word) |> 
  filter(n>1) #Filtering for words/bigrams per document, rather than per corpus

tidy_docs <- tidy_docs |> # filter words that are just numbers
  filter(! stringr::str_detect(tidy_docs$word, "^[0-9]+$"))

# append observation level data 
colnames(tidy_docs)[1:2] <- c("document", "term")


# turn a tidy tbl into a sparse dgCMatrix 
# note tidylda has support for several document term matrix formats
d <- tidy_docs |> 
  cast_sparse(document, term, n)

# let's split the documents into two groups to demonstrate predictions and updates
d1 <- d[1:50, ]

d2 <- d[51:nrow(d), ]

# make sure we have different vocabulary for each data set to simulate the "real world"
# where you get new tokens coming in over time
d1 <- d1[, colSums(d1) > 0]

d2 <- d2[, colSums(d2) > 0]

### fit an intial model and inspect it ----
set.seed(123)

lda <- tidylda(
  data = d1,
  k = 10,
  iterations = 200, 
  burnin = 175,
  alpha = 0.1, # also accepts vector inputs
  eta = 0.05, # also accepts vector or matrix inputs
  optimize_alpha = FALSE, # experimental
  calc_likelihood = TRUE,
  calc_r2 = TRUE, # see https://arxiv.org/abs/1911.11061
  return_data = FALSE
)

# did the model converge?
# there are actual test stats for this, but should look like "yes"
qplot(x = iteration, y = log_likelihood, data = lda$log_likelihood, geom = "line") + 
    ggtitle("Checking model convergence")

# look at the model overall
glance(lda)

print(lda)

# it comes with its own summary matrix that's printed out with print(), above
lda$summary


# inspect the individual matrices
tidy_theta <- tidy(lda, matrix = "theta")

tidy_theta

tidy_beta <- tidy(lda, matrix = "beta")

tidy_beta

tidy_lambda <- tidy(lda, matrix = "lambda")

tidy_lambda

# append observation-level data
augmented_docs <- augment(lda, data = tidy_docs)

augmented_docs

### predictions on held out data ---
# two methods: gibbs is cleaner and more technically correct in the bayesian sense
p_gibbs <- predict(lda, new_data = d2[1, ], iterations = 100, burnin = 75)

# dot is faster, less prone to error (e.g. underflow), noisier, and frequentist
p_dot <- predict(lda, new_data = d2[1, ], method = "dot")

# pull both together into a plot to compare
tibble(topic = 1:ncol(p_gibbs), gibbs = p_gibbs[1,], dot = p_dot[1, ]) |>
  pivot_longer(cols = gibbs:dot, names_to = "type") |>
  ggplot() + 
  geom_bar(mapping = aes(x = topic, y = value, group = type, fill = type), 
           stat = "identity", position="dodge") +
  scale_x_continuous(breaks = 1:10, labels = 1:10) + 
  ggtitle("Gibbs predictions vs. dot product predictions")

### Augment as an implicit prediction using the 'dot' method ----
# Aggregating over terms results in a distribution of topics over documents
# roughly equivalent to using the "dot" method of predictions.
augment_predict <- 
  augment(lda, tidy_docs, "prob") |>
  group_by(document) |> 
  select(-c(document, term)) |> 
  summarise_all(function(x) sum(x, na.rm = T))

# reformat for easy plotting
augment_predict <- 
  as_tibble(t(augment_predict[, -c(1,2)]), .name_repair = "minimal")

colnames(augment_predict) <- unique(tidy_docs$document)

augment_predict$topic <- 1:nrow(augment_predict) |> as.factor()

compare_mat <- 
  augment_predict |>
  select(
    topic,
    augment = matches(rownames(d2)[1])
  ) |>
  mutate(
    augment = augment / sum(augment), # normalize to sum to 1
    dot = p_dot[1, ]
  ) |>
  pivot_longer(cols = c(augment, dot))

ggplot(compare_mat) + 
  geom_bar(aes(y = value, x = topic, group = name, fill = name), 
           stat = "identity", position = "dodge") +
  labs(title = "Prediction using 'augment' vs 'predict(..., method = \"dot\")'")

# Not shown: aggregating over documents results in recovering the "tidy" lambda.

### updating the model ----
# now that you have new documents, maybe you want to fold them into the model?
lda2 <- refit(
  object = lda, 
  new_data = d, # save me the trouble of manually-combining these by just using d
  iterations = 200, 
  burnin = 175,
  calc_likelihood = TRUE,
  calc_r2 = TRUE
)

# we can do similar analyses
# did the model converge?
qplot(x = iteration, y = log_likelihood, data = lda2$log_likelihood, geom = "line") +
  ggtitle("Checking model convergence")

# look at the model overall
glance(lda2)

print(lda2)


# how does that compare to the old model?
print(lda)

