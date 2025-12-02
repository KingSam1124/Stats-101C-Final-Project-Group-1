############################################################
## Stats 101C – Aluminum ColdRoll – CatBoost model (R)
############################################################

rm(list = ls())

## ---- Libraries ----
library(data.table)
library(catboost)

set.seed(777)

## ---- 1. Read data ----
train_data <- fread("data/aluminum_coldRoll_train.csv")
test_data  <- fread("data/aluminum_coldRoll_testNoY.csv")

id_col     <- "ID"
target_col <- "y_passXtremeDurability"

cat("Train shape:", dim(train_data), "\n")
cat("Test  shape:", dim(test_data),  "\n")

## --- 1.5  Make sure there are no character (text) columns ----
char_cols <- names(which(sapply(train_data, is.character)))
if (length(char_cols) > 0) {
  cat("Converting character columns to integer codes:\n")
  print(char_cols)
  
  for (cc in char_cols) {
    lev <- sort(unique(train_data[[cc]]))
    # same levels in train/test
    train_data[[cc]] <- as.integer(factor(train_data[[cc]], levels = lev))
    test_data[[cc]]  <- as.integer(factor(test_data[[cc]],  levels = lev))
  }
}


## ---- 2. Split into X / y ----
y       <- train_data[[target_col]]

X       <- train_data[, !c(id_col, target_col), with = FALSE]
X_test  <- test_data[, !id_col, with = FALSE]
test_id <- test_data[[id_col]]

feature_cols <- names(X)
cat("Number of features:", length(feature_cols), "\n")
print(feature_cols)

## ---- 3. Train / validation split ----
n <- nrow(X)
val_frac <- 0.20
val_size <- floor(n * val_frac)

set.seed(777)
val_frac <- 0.2
val_size <- as.integer(n * val_frac)
val_idx  <- sample(seq_len(n), size = val_size)
train_idx <- setdiff(seq_len(n), val_idx)

X_train <- X[train_idx, ]
X_val   <- X[val_idx, ]
y_train <- y[train_idx]
y_val   <- y[val_idx]

cat("Train rows:", nrow(X_train), "  Val rows:", nrow(X_val), "\n")

## ---- 4. Build CatBoost pools ----
train_pool <- catboost.load_pool(
  data  = X_train,
  label = y_train
)

val_pool <- catboost.load_pool(
  data  = X_val,
  label = y_val
)

test_pool <- catboost.load_pool(
  data = X_test
)

## ---- 5. CatBoost parameters ----
params <- list(
  loss_function = "Logloss",
  eval_metric   = "Logloss",
  iterations    = 3000,
  learning_rate = 0.03,
  depth         = 6,
  l2_leaf_reg   = 3,
  random_seed   = 777,
  border_count  = 128,
  od_type       = "Iter",   # early stopping on iterations
  od_wait       = 100,
  verbose       = 100
)

## ---- 6. Train model with early stopping ----
cat("Training CatBoost model...\n")
model <- catboost.train(
  learn_pool = train_pool,
  test_pool  = val_pool,
  params     = params
)

cat("Model trained.\n")

## Optional: quick sanity check on validation
pred_val <- catboost.predict(
  model,
  val_pool,
  prediction_type = "Probability"
)

cat("Validation preds range: ",
    range(as.numeric(pred_val)), "\n")

## ---- 7. Predict on test & build submission ----
cat("Predicting on test set...\n")

pred_test <- catboost.predict(
  model,
  test_pool,
  prediction_type = "Probability"
)

pred_vec <- as.numeric(pred_test)

## Safety: replace any NA / Inf and clip to (0,1)
bad <- !is.finite(pred_vec)
if (any(bad)) {
  cat("Found", sum(bad), "non-finite predictions, replacing with 0.5\n")
  pred_vec[bad] <- 0.5
}

eps <- 1e-6
pred_vec[pred_vec < eps]       <- eps
pred_vec[pred_vec > 1 - eps]   <- 1 - eps

submission <- data.table(
  ID                    = test_id,
  y_passXtremeDurability = pred_vec
)

dir.create("submissions", showWarnings = FALSE)

out_path <- "submissions/submission_catboost_R.csv"
fwrite(submission, out_path, row.names = FALSE)

cat("Done! Submission file saved to", out_path, "\n")
