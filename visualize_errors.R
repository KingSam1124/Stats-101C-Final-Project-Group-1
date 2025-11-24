# Add the local library to the library paths
.libPaths(c("../R_libs", .libPaths()))

# Load necessary libraries
library(data.table)

# --- Visualization for GLM ---
tryCatch({
  glm_log <- fread("results/glm_cv_log.csv")
  png("results/plots/glm_cv_error.png", width = 800, height = 600)
  
  # Plot mean cross-validated error vs. log(lambda)
  plot(log(glm_log$lambda), glm_log$cvm, type = 'l', col = 'blue', lwd = 2,
       xlab = "Log(Lambda)", ylab = "Binomial Deviance", main = "GLM Cross-Validation Error")
  
  # Add lines for standard deviation
  lines(log(glm_log$lambda), glm_log$cvm + glm_log$cvsd, col = 'red', lty = 2)
  lines(log(glm_log$lambda), glm_log$cvm - glm_log$cvsd, col = 'red', lty = 2)
  
  # Add vertical lines for lambda.min and lambda.1se
  cv_model_summary <- summary(glm_log) # A bit of a hack to find min lambda index
  lambda_min_val <- glm_log$lambda[which.min(glm_log$cvm)]
  abline(v = log(lambda_min_val), col = "blue", lty = 3, lwd = 2)
  
  legend("topright", legend = c("Mean CV Error", "Std. Dev.", "log(lambda.min)"), 
         col = c("blue", "red", "blue"), lty = c(1, 2, 3), lwd = 2)
         
  dev.off()
  cat("Generated GLM error plot: results/plots/glm_cv_error.png\n")
}, error = function(e) {
  cat("Could not generate GLM plot. Did you run model_glm.R first?\nError: ", e$message, "\n")
})


# --- Visualization for XGBoost ---
tryCatch({
  xgb_log <- fread("results/xgb_cv_log.csv")
  png("results/plots/xgb_cv_error.png", width = 800, height = 600)
  
  # Plot training and validation logloss
  plot(xgb_log$iter, xgb_log$train_logloss_mean, type = 'l', col = 'blue', lwd = 2,
       ylim = range(c(xgb_log$train_logloss_mean, xgb_log$test_logloss_mean)),
       xlab = "Boosting Iteration", ylab = "LogLoss", main = "XGBoost Cross-Validation Performance")
  lines(xgb_log$iter, xgb_log$test_logloss_mean, type = 'l', col = 'red', lwd = 2)
  
  legend("topright", legend = c("Training LogLoss", "Validation LogLoss"), 
         col = c("blue", "red"), lty = 1, lwd = 2)
         
  dev.off()
  cat("Generated XGBoost error plot: results/plots/xgb_cv_error.png\n")
}, error = function(e) {
  cat("Could not generate XGBoost plot. Did you run model_xgb.R first?\nError: ", e$message, "\n")
})


# --- Visualization for Neural Network ---
tryCatch({
  nn_log <- fread("results/nn_training_history.csv")
  # Add an epoch column
  nn_log[, epoch := 1:.N]
  
  # Plot for Loss
  png("results/plots/nn_loss.png", width = 800, height = 600)
  plot(nn_log$epoch, nn_log$loss, type = 'l', col = 'blue', lwd = 2,
       ylim = range(c(nn_log$loss, nn_log$val_loss)),
       xlab = "Epoch", ylab = "Loss (Binary Crossentropy)", main = "Neural Network Training vs. Validation Loss")
  lines(nn_log$epoch, nn_log$val_loss, type = 'l', col = 'red', lwd = 2)
  legend("topright", legend = c("Training Loss", "Validation Loss"), 
         col = c("blue", "red"), lty = 1, lwd = 2)
  dev.off()
  cat("Generated Neural Network loss plot: results/plots/nn_loss.png\n")

  # Plot for Accuracy
  png("results/plots/nn_accuracy.png", width = 800, height = 600)
  plot(nn_log$epoch, nn_log$accuracy, type = 'l', col = 'blue', lwd = 2,
       ylim = range(c(nn_log$accuracy, nn_log$val_accuracy)),
       xlab = "Epoch", ylab = "Accuracy", main = "Neural Network Training vs. Validation Accuracy")
  lines(nn_log$epoch, nn_log$val_accuracy, type = 'l', col = 'red', lwd = 2)
  legend("bottomright", legend = c("Training Accuracy", "Validation Accuracy"), 
         col = c("blue", "red"), lty = 1, lwd = 2)
  dev.off()
  cat("Generated Neural Network accuracy plot: results/plots/nn_accuracy.png\n")

}, error = function(e) {
  cat("Could not generate Neural Network plots. Did you run model_nn.R first?\nError: ", e$message, "\n")
})
