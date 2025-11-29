
# Add the local library to the library paths

.libPaths(c("../R_libs", .libPaths()))

# Load necessary libraries

library(data.table)
library(xgboost)

# Load the training data

train_data <- fread("aluminum_coldRoll_train.csv")
test_data <- fread("aluminum_coldRoll_testNoY.csv")

# Combine data 

test_id <- test_data$ID
test_data[, ID := NULL]
train_data[, ID := NULL]

n_train <- nrow(train_data)

y_train <- train_data$y_passXtremeDurability
if (is.factor(y_train) || is.character(y_train)) {
  y_train <- as.numeric(as.character(y_train))
}
y_train <- as.numeric(y_train)

train_data[, y_passXtremeDurability := NULL]


# Feature Engineering 

temp_map <- c("low" = 0, "med" = 1, "high" = 2)

# Ordinal

train_data[, cutTemp_ord  := temp_map[cutTemp]]
test_data[,  cutTemp_ord  := temp_map[cutTemp]]

train_data[, rollTemp_ord := temp_map[rollTemp]]
test_data[,  rollTemp_ord := temp_map[rollTemp]]

# Temperature combo

train_data[, temp_combo := paste(cutTemp, rollTemp, sep = "_")]
test_data[,  temp_combo := paste(cutTemp, rollTemp, sep = "_")]

# Alloy family

train_data[, alloy_family := substr(alloy, 1, 2)]
test_data[,  alloy_family := substr(alloy, 1, 2)]

# Pressure features

train_data[, pressure_ratio := secondPassRollPressure / firstPassRollPressure]
test_data[,  pressure_ratio := secondPassRollPressure / firstPassRollPressure]

train_data[!is.finite(pressure_ratio), pressure_ratio := NA_real_]
test_data[!is.finite(pressure_ratio),  pressure_ratio := NA_real_]

train_data[, pressure_diff := secondPassRollPressure - firstPassRollPressure]
test_data[,  pressure_diff := secondPassRollPressure - firstPassRollPressure]

train_data[, second_higher := as.integer(secondPassRollPressure > firstPassRollPressure)]
test_data[,  second_higher := as.integer(secondPassRollPressure > firstPassRollPressure)]

# Machine restart binary

train_data[, machineRestart_bin := as.integer(machineRestart == "yes")]
test_data[,  machineRestart_bin := as.integer(machineRestart == "yes")]

drop_cols <- c("cutTemp", "rollTemp", "machineRestart", "temp_combo")
train_data[, (drop_cols) := NULL]
test_data[,  (drop_cols) := NULL]

train_data[, pressure_norm_diff := 
             (secondPassRollPressure - firstPassRollPressure) /
             (secondPassRollPressure + firstPassRollPressure)]
test_data[,  pressure_norm_diff := 
            (secondPassRollPressure - firstPassRollPressure) /
            (secondPassRollPressure + firstPassRollPressure)]

train_data[!is.finite(pressure_norm_diff), pressure_norm_diff := 0]
test_data[!is.finite(pressure_norm_diff),  pressure_norm_diff := 0]

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


# Setting parameters

cat("Using tuned XGBoost parameters.\n")
params <- list(
  objective = "binary:logistic",
  eval_metric = "logloss",
  eta = 0.03,              
  max_depth = 4,          
  subsample = 0.8,
  colsample_bytree = 0.8,
  gamma = 0.05,            
  min_child_weight = 2     
)

# xgb.cv with more rounds and earlier stopping

cat("Starting XGBoost cross-validation with 2000 rounds...\n")
xgb_cv <- xgb.cv(
  params = params,
  data = dtrain,
  nrounds = 2000,          
  nfold = 5,
  showsd = TRUE,
  stratified = TRUE,
  print_every_n = 100,     
  early_stopping_rounds = 120, 
  maximize = FALSE
)

# Save the cross-validation evaluation log for the tuned model
write.csv(xgb_cv$evaluation_log, "results/xgb_tuned_cv_log.csv", row.names = FALSE)

best_nrounds <- xgb_cv$best_iteration
best_score <- xgb_cv$evaluation_log[best_nrounds]$test_logloss_mean

# Printing the best score

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

predictions <- pmin(pmax(predictions, 1e-6), 1 - 1e-6)

# Create the submission file
submission <- data.table(
  ID = test_id,
  y_passXtremeDurability = predictions
)

# Save the submission file for the tuned model
write.csv(submission, "submissions/submission_xgb_tuned.csv", row.names = FALSE, quote = FALSE)
cat("Tuned submission file saved to: submissions/submission_xgb_tuned.csv\n")