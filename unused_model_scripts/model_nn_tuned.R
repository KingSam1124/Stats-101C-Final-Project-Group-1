# Add the local library to the library paths
.libPaths(c("../R_libs", .libPaths()))

# Load necessary libraries
library(data.table)
library(keras)

# --- 1. Data Preparation ---
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
    if (max_val > min_val) {
        combined_data[[col]] <- (combined_data[[col]] - min_val) / (max_val - min_val)
    }
}

# Create a model matrix
full_matrix <- as.matrix(combined_data)

# Separate back into training and test matrices
train_matrix_full <- full_matrix[1:n_train, ]
test_matrix <- full_matrix[(n_train + 1):nrow(full_matrix), ]

# Impute NAs with column means from the full training set
for(i in 1:ncol(train_matrix_full)){
  mean_val <- mean(train_matrix_full[,i], na.rm = TRUE)
  train_matrix_full[is.na(train_matrix_full[,i]), i] <- mean_val
  test_matrix[is.na(test_matrix[,i]), i] <- mean_val # Use train mean for test set
}

# Manually split the training data for use with callbacks
set.seed(42)
val_indices <- sample(1:nrow(train_matrix_full), 0.2 * nrow(train_matrix_full))
train_indices <- setdiff(1:nrow(train_matrix_full), val_indices)

x_train <- train_matrix_full[train_indices, ]
y_train_split <- y_train[train_indices]
x_val <- train_matrix_full[val_indices, ]
y_val_split <- y_train[val_indices]


# --- 2. Model Definition ---
# A slightly larger model with L2 regularization
model <- keras_model_sequential() %>%
  layer_dense(units = 256, activation = "relu", input_shape = ncol(x_train),
              kernel_regularizer = regularizer_l2(l = 0.001)) %>%
  layer_batch_normalization() %>%
  layer_dropout(rate = 0.4) %>%
  layer_dense(units = 128, activation = "relu",
              kernel_regularizer = regularizer_l2(l = 0.001)) %>%
  layer_batch_normalization() %>%
  layer_dropout(rate = 0.3) %>%
  layer_dense(units = 64, activation = "relu",
              kernel_regularizer = regularizer_l2(l = 0.001)) %>%
  layer_batch_normalization() %>%
  layer_dropout(rate = 0.2) %>%
  layer_dense(units = 1, activation = "sigmoid")

# Compile the model with a tuned optimizer
model %>%
  compile(
  loss = "binary_crossentropy",
  optimizer = optimizer_adam(learning_rate = 0.0005),
  metrics = c("accuracy")
)


# --- 3. Callbacks for Robust Training ---
# Stop training when validation loss doesn't improve
early_stopping <- callback_early_stopping(patience = 15, restore_best_weights = TRUE)
# Save the best model found during training
model_checkpoint <- callback_model_checkpoint("best_nn_model.h5", save_best_only = TRUE, monitor = "val_loss")
# Reduce learning rate if training plateaus
reduce_lr <- callback_reduce_lr_on_plateau(monitor = "val_loss", factor = 0.2, patience = 5)


# --- 4. Model Training ---
cat("Starting tuned Neural Network training with callbacks...\n")
history <- model %>%
  fit(
  x_train,
  y_train_split,
  epochs = 150,
  batch_size = 128,
  validation_data = list(x_val, y_val_split),
  callbacks = list(early_stopping, model_checkpoint, reduce_lr),
  verbose = 2 # Use verbose=2 for one line per epoch
)

# Save the training history
write.csv(as.data.frame(history$metrics), "results/nn_tuned_training_history.csv", row.names = FALSE)
cat("\nTuned NN training complete. Best model saved to best_nn_model.h5\n")


# --- 5. Prediction ---
# Load the best performing model from training to make predictions
cat("Loading best model from training to make predictions...\n")
best_model <- load_model_hdf5("best_nn_model.h5")
predictions <- best_model %>%
  predict(test_matrix)

# Create submission file
submission <- data.table(ID = test_id, y_passXtremeDurability = predictions[,1])
write.csv(submission, "submissions/submission_nn_tuned.csv", row.names = FALSE, quote = FALSE)
cat("Tuned NN submission file saved to: submissions/submission_nn_tuned.csv\n")
