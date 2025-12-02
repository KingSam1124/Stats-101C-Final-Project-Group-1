# Add the local library to the library paths
.libPaths(c("../R_libs", .libPaths()))

# Load necessary libraries
library(data.table)
library(keras)

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

# Scale numeric features
numeric_cols <- which(sapply(combined_data, is.numeric))
for (col in numeric_cols) {
    max_val <- max(combined_data[[col]], na.rm = TRUE)
    min_val <- min(combined_data[[col]], na.rm = TRUE)
    combined_data[[col]] <- (combined_data[[col]] - min_val) / (max_val - min_val)
}

# Create a model matrix
full_matrix <- as.matrix(combined_data)

# Separate back into training and test matrices
train_matrix_new <- full_matrix[1:n_train, ]
test_matrix_new <- full_matrix[(n_train + 1):nrow(full_matrix), ]

# Impute NAs with column means
for(i in 1:ncol(train_matrix_new)){
  train_matrix_new[is.na(train_matrix_new[,i]), i] <- mean(train_matrix_new[,i], na.rm = TRUE)
}
for(i in 1:ncol(test_matrix_new)){
  test_matrix_new[is.na(test_matrix_new[,i]), i] <- mean(test_matrix_new[,i], na.rm = TRUE) # Use train mean for test
}

# Define the model
model <- keras_model_sequential()
model %>%
  layer_dense(units = 128, activation = "relu", input_shape = ncol(train_matrix_new)) %>%
  layer_dropout(rate = 0.3) %>%
  layer_dense(units = 64, activation = "relu") %>%
  layer_dropout(rate = 0.2) %>%
  layer_dense(units = 1, activation = "sigmoid")

# Compile the model
model %>% compile(
  loss = "binary_crossentropy",
  optimizer = "adam",
  metrics = c("accuracy")
)

# Train the model
cat("Starting Neural Network training...\n")
history <- model %>% fit(
  train_matrix_new,
  y_train,
  epochs = 20,
  batch_size = 128,
  validation_split = 0.2,
  verbose = 1
)

# Save the training history
write.csv(as.data.frame(history$metrics), "results/nn_training_history.csv", row.names = FALSE)

# Make predictions
predictions <- model %>% predict(test_matrix_new)

# Create submission file
submission <- data.table(ID = test_id, y_passXtremeDurability = predictions[,1])
write.csv(submission, "submissions/submission_nn.csv", row.names = FALSE, quote = FALSE)
