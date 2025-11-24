# Add the local library to the library paths
.libPaths(c("../R_libs", .libPaths()))

# Load necessary libraries
library(data.table)
library(xgboost)

# Load the training data
train_data <- fread("data/aluminum_coldRoll_train.csv")
test_data <- fread("data/aluminum_coldRoll_testNoY.csv")

# Combine data for consistent feature engineering
test_id <- test_data$ID
test_data[, ID := NULL]
train_data[, ID := NULL]

n_train <- nrow(train_data)
y_train <- as.numeric(train_data$y_passXtremeDurability)
train_data[, y_passXtremeDurability := NULL]

combined_data <- rbind(train_data, test_data, fill=TRUE)

# Convert character columns to factors and then to numeric
for (col in names(combined_data)) {
  if (is.character(combined_data[[col]]) || is.factor(combined_data[[col]])) {
    combined_data[[col]] <- as.integer(as.factor(combined_data[[col]]))
  }
}

# Create a model matrix
full_matrix <- as.matrix(combined_data)

# Separate back into training and test matrices
train_matrix_new <- full_matrix[1:n_train, ]
test_matrix_new <- full_matrix[(n_train + 1):nrow(full_matrix), ]


# Create xgb.DMatrix objects
dtrain <- xgb.DMatrix(data = train_matrix_new, label = y_train)
dtest <- xgb.DMatrix(data = test_matrix_new)


# Set tuned xgboost parameters to achieve a lower score
cat("Using tuned XGBoost parameters.\n")
params <- list(
  objective = "binary:logistic",
  eval_metric = "logloss",
  eta = 0.02,              # Lower learning rate
  max_depth = 5,           # Reduced max_depth to make model more robust
  subsample = 0.8,
  colsample_bytree = 0.8,
  gamma = 0.1,             # Added gamma for regularization
  min_child_weight = 1     # Added min_child_weight for regularization
)

# Use xgb.cv with more rounds and earlier stopping
cat("Starting XGBoost cross-validation with 2000 rounds...\n")
xgb_cv <- xgb.cv(
  params = params,
  data = dtrain,
  nrounds = 2000,          # Increased number of rounds
  nfold = 5,
  showsd = TRUE,
  stratified = TRUE,
  print_every_n = 100,     # Print progress every 100 rounds
  early_stopping_rounds = 50, # More patient early stopping
  maximize = FALSE
)

# Save the cross-validation evaluation log for the tuned model
write.csv(xgb_cv$evaluation_log, "results/xgb_tuned_cv_log.csv", row.names = FALSE)

best_nrounds <- xgb_cv$best_iteration
best_score <- xgb_cv$evaluation_log[best_nrounds]$test_logloss_mean

# Print the best score achieved
cat("\n--------------------------------------------------\n")
cat("XGBoost Cross-Validation Complete.\n")
cat("Best CV LogLoss Score:", round(best_score, 5), "at round", best_nrounds, "\n")
cat("--------------------------------------------------\n\n")

# Train the final xgboost model with the best number of rounds
cat("Training final model with best number of rounds...\n")
xgb_model <- xgboost(
  params = params,
  data = dtrain,
  nrounds = best_nrounds,
  print_every_n = 100
)
cat("Final model training complete.\n")

# Make predictions on the test data
predictions <- predict(xgb_model, dtest)

# Create the submission file
submission <- data.table(ID = test_id, y_passXtremeDurability = round(predictions, 4))

# Save the submission file for the tuned model
write.csv(submission, "submissions/submission_xgb_tuned.csv", row.names = FALSE, quote = FALSE)
cat("Tuned submission file saved to: submissions/submission_xgb_tuned.csv\n")
