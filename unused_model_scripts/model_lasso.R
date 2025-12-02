.libPaths(c("../R_libs", .libPaths()))

library(data.table)
library(glmnet)

set.seed(1)

# Load the training data
train_data <- fread("Documents/Stats-101C-Final-Project-Group-1/data/aluminum_coldRoll_train.csv")
test_data <- fread("Documents/Stats-101C-Final-Project-Group-1/data/aluminum_coldRoll_testNoY.csv")

# Prep data
test_id <- test_data$ID
test_data[, ID := NULL]
train_data[, ID := NULL]

n_train <- nrow(train_data)

y_train <- train_data$y_passXtremeDurability

# If response is a factor/character, convert to 0/1 numeric
if (is.factor(y_train) || is.character(y_train)) {
  y_train <- as.numeric(as.character(y_train))
}
y_train <- as.numeric(y_train)

train_data[, y_passXtremeDurability := NULL]


# Combine data for consistent feature engineering
combined_data <- rbind(train_data, test_data, fill=TRUE)

# Convert character columns to factors - now all data is numeric or factor
for (col in names(combined_data)) {
  if (is.character(combined_data[[col]])) {
    combined_data[[col]] <- as.factor(combined_data[[col]])
  }
}

# Separate back into training and test sets
combined_new <- model.matrix(~ . - 1, data = combined_data) # Dummy encoding

train_new <- combined_new[1:n_train, , drop = FALSE]
test_new <- combined_new[(n_train + 1):nrow(combined_new), , drop = FALSE]

# Train lasso model
grid <- 10^seq(10, -2, length = 100)
lasso_model <- cv.glmnet(x = train_new, y = y_train, alpha = 1, lambda = grid)
summary(lasso_model)  

cat("Lasso model complete.\n")

# Predict probabilities
best_lambda <- lasso_model$lambda.min # Select best lambda using CV
print(best_lambda)

predictions <- predict(lasso_model, s = best_lambda, newx = test_new, type = "response")
predictions <- pmin(pmax(predictions, 1e-6), 1 - 1e-6)

coef(lasso_model, s = best_lambda)

# Building submission file
submission <- data.table(ID = test_id, y_passXtremeDurability = predictions)
write.csv(submission, "Documents/Stats-101C-Final-Project-Group-1/submissions/submission_lasso.csv", row.names = FALSE, quote = FALSE)
cat("Submission file saved to:", "submissions/submission_lasso.csv\n")
