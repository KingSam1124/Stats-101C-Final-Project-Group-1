# Regression Model: 

# Setup 

.libPaths(c("../R_libs", .libPaths()))

library(data.table)

set.seed(1)


# Loading data

train_data <- fread("aluminum_coldRoll_train.csv")
test_data  <- fread("aluminum_coldRoll_testNoY.csv")


test_id <- test_data$ID
test_data[, ID := NULL]
train_data[, ID := NULL]


# Target

y_train <- train_data$y_passXtremeDurability

# if it's factor/character, convert to 0/1 numeric
if (is.factor(y_train) || is.character(y_train)) {
  y_train <- as.numeric(as.character(y_train))
}
y_train <- as.numeric(y_train)

# Remove target from feature frame
train_data[, y_passXtremeDurability := NULL]

n_train <- nrow(train_data)


combined_data <- rbind(train_data, test_data, fill = TRUE)

# Convert character columns to factors (needed for model.matrix)
for (col in names(combined_data)) {
  if (is.character(combined_data[[col]])) {
    combined_data[[col]] <- as.factor(combined_data[[col]])
  }
}

X_full <- model.matrix(~ . - 1, data = combined_data)

X_train <- X_full[1:n_train, , drop = FALSE]
X_test  <- X_full[(n_train + 1):nrow(X_full), , drop = FALSE]

train_df <- as.data.frame(X_train)
test_df  <- as.data.frame(X_test)


# Fitting logistic regression 

cat("Fitting logistic regression model...\n")

logit_model <- glm(
  y_train ~ .,
  data   = train_df,
  family = binomial(link = "logit")
)

summary(logit_model)  

cat("Logistic regression fit complete.\n")


# Predict probabilities

predictions <- predict(logit_model, newdata = test_df, type = "response")

predictions <- pmin(pmax(predictions, 1e-6), 1 - 1e-6)


# Building submission file

submission <- data.table(
  ID = test_id,
  y_passXtremeDurability = predictions
)

dir.create("submissions", showWarnings = FALSE)
fwrite(submission, "submissions/submission_logistic_regression.csv",
       row.names = FALSE, quote = FALSE)

cat("Logistic regression submission saved to:",
    "submissions/submission_logistic_regression.csv\n")
