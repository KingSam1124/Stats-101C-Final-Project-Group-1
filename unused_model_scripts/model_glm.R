# Add the local library to the library paths
.libPaths(c("../R_libs", .libPaths()))

# Load necessary libraries
library(data.table)
library(glmnet)

# Load the training data
train_data <- fread("data/aluminum_coldRoll_train.csv")
test_data <- fread("data/aluminum_coldRoll_testNoY.csv")

# Combine data for consistent feature engineering
test_id <- test_data$ID # Store ID for submission
test_data[, ID := NULL]
train_data[, ID := NULL]

# Store original row counts
n_train <- nrow(train_data)
n_test <- nrow(test_data)

# Separate target variable before combining
y_train <- as.numeric(train_data$y_passXtremeDurability)
train_data[, y_passXtremeDurability := NULL] # Remove target from features

# Combined data for consistent preprocessing
combined_data <- rbind(train_data, test_data, fill=TRUE) # Use fill=TRUE to handle missing columns if any

# Convert character columns to factors
for (col in names(combined_data)) {
  if (is.character(combined_data[[col]])) {
    combined_data[[col]] <- as.factor(combined_data[[col]])
  }
}

# Feature Engineering
combined_data[, firstPassRollPressure_sq := firstPassRollPressure^2]
combined_data[, secondPassRollPressure_sq := secondPassRollPressure^2]
combined_data[, contourDefNdx_sq := contourDefNdx^2]
combined_data[, clearPassNdx_sq := clearPassNdx^2]

# Ensure `drop = FALSE` in interaction to keep all levels even if missing in one subset
combined_data[, cutTemp_rollTemp_interaction := interaction(cutTemp, rollTemp, sep = "_", drop = FALSE)]
combined_data[, first_second_pressure_interaction := firstPassRollPressure * secondPassRollPressure]
combined_data[, microChipping_rollTemp_interaction := interaction(topEdgeMicroChipping, rollTemp, sep = "_", drop = FALSE)]

# Define the formula for features only
# Use . to include all current columns, then remove newly created interactions if they're redundant
feature_cols <- names(combined_data)
formula_str <- paste0("~ . - 1 +
  firstPassRollPressure_sq + secondPassRollPressure_sq +
  contourDefNdx_sq + clearPassNdx_sq +
  cutTemp:rollTemp + firstPassRollPressure:secondPassRollPressure + topEdgeMicroChipping:rollTemp")

# Create model matrix. na.action=na.pass to retain all rows, including those with NA if any.
# Any NA values in features will be handled by glmnet's default behavior, or can be imputed if necessary.
full_matrix <- model.matrix(as.formula(formula_str), data = combined_data, na.action = na.pass)

# Separate back into training and test matrices
train_matrix_new <- full_matrix[1:n_train, ]
test_matrix_new <- full_matrix[(n_train + 1):nrow(full_matrix), ]


# Train a logistic regression model using glmnet
# Use cross-validation to find the optimal lambda
cat("Starting GLM model training (cv.glmnet)...\n")
cv_model <- cv.glmnet(train_matrix_new, y_train, family = "binomial")
cat("GLM model training complete.\n")

# Save the cross-validation results
cv_results <- data.frame(
  lambda = cv_model$lambda,
  cvm = cv_model$cvm, # Mean cross-validated error
  cvsd = cv_model$cvsd # Standard deviation of error
)
write.csv(cv_results, "results/glm_cv_log.csv", row.names = FALSE)

# Print the cross-validation results
print(cv_model)

# Make predictions on the test data
predictions <- predict(cv_model, newx = test_matrix_new, s = "lambda.min", type = "response")

# Create the submission file
submission <- data.table(ID = test_id, y_passXtremeDurability = round(predictions[,1], 4))

# Save the submission file
write.csv(submission, "submissions/submission_glm.csv", row.names = FALSE, quote = FALSE)
