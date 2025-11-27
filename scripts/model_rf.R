# Add the local library to the library paths
.libPaths(c("../R_libs", .libPaths()))

# Load necessary libraries
library(data.table)
library(randomForest)
set.seed(123)

# Load the training data
train_data <- fread("Documents/Stats-101C-Final-Project-Group-1/data/aluminum_coldRoll_train.csv")
test_data <- fread("Documents/Stats-101C-Final-Project-Group-1/data/aluminum_coldRoll_testNoY.csv")

# Combine data for consistent feature engineering
test_id <- test_data$ID
test_data[, ID := NULL]
train_data[, ID := NULL]

n_train <- nrow(train_data)
y_train <- as.factor(train_data$y_passXtremeDurability)
train_data[, y_passXtremeDurability := NULL]

combined_data <- rbind(train_data, test_data, fill=TRUE)

# Convert character columns to factors
for (col in names(combined_data)) {
  if (is.character(combined_data[[col]])) {
    combined_data[[col]] <- as.factor(combined_data[[col]])
  }
}

# Separate back into training and test matrices
train_matrix_new <- combined_data[1:n_train, ]
test_matrix_new <- combined_data[(n_train + 1):nrow(combined_data), ]

# Train a random forest model
rf_model <- randomForest(
  x = train_matrix_new,
  y = y_train,
  ntree = 500,
  mtry = 10, # Bagging
  importance = TRUE,
  do.trace = 50
)

# Perform cross validation for mtry selection?

# Make predictions on the test data
predictions <- predict(rf_model, test_matrix_new, type = "prob")[,"1"]

# Create the submission file
submission <- data.table(ID = test_id, y_passXtremeDurability = predictions)

# Save the submission file for the tuned model
write.csv(submission, "Documents/Stats-101C-Final-Project-Group-1/submissions/submission_rf.csv", row.names = FALSE, quote = FALSE)
cat("Tuned submission file saved to: submissions/submission_rf.csv\n")
