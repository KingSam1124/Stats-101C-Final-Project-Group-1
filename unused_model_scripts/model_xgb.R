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
# This is a different approach than one-hot encoding, and can be effective for tree-based models
for (col in names(combined_data)) {
  if (is.character(combined_data[[col]]) || is.factor(combined_data[[col]])) {
    combined_data[[col]] <- as.integer(as.factor(combined_data[[col]]))
  }
}

# Create a model matrix. xgboost can handle numeric matrices.
# This approach avoids the complexity and potential errors of model.matrix with many factors.
full_matrix <- as.matrix(combined_data)

# Separate back into training and test matrices
train_matrix_new <- full_matrix[1:n_train, ]
test_matrix_new <- full_matrix[(n_train + 1):nrow(full_matrix), ]


# Create xgb.DMatrix objects
dtrain <- xgb.DMatrix(data = train_matrix_new, label = y_train)
dtest <- xgb.DMatrix(data = test_matrix_new)


# Set xgboost parameters
# Using a set of reasonable defaults for a first attempt
params <- list(
  objective = "binary:logistic",
  eval_metric = "logloss",
  eta = 0.05,
  max_depth = 6,
  subsample = 0.8,
  colsample_bytree = 0.8
)

# Use xgb.cv to find the best number of rounds
# This helps prevent overfitting and finds a good stopping point
xgb_cv <- xgb.cv(
  params = params,
  data = dtrain,
  nrounds = 500,
  nfold = 5,
  showsd = TRUE,
  stratified = TRUE,
  print_every_n = 10,
  early_stopping_rounds = 20,
  maximize = FALSE
)

# Save the cross-validation evaluation log
write.csv(xgb_cv$evaluation_log, "results/xgb_cv_log.csv", row.names = FALSE)

best_nrounds <- xgb_cv$best_iteration

# Train the final xgboost model with the best number of rounds
xgb_model <- xgboost(
  params = params,
  data = dtrain,
  nrounds = best_nrounds,
  print_every_n = 50
)

# Make predictions on the test data
predictions <- predict(xgb_model, dtest)

# Create the submission file
submission <- data.table(ID = test_id, y_passXtremeDurability = round(predictions, 4))

# Save the submission file
write.csv(submission, "submissions/submission_xgb.csv", row.names = FALSE, quote = FALSE)