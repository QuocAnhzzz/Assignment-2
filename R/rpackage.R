rm(list = ls())
ls()

# Utils Functions
calculate_cm <- function(predict_model) {
  if (is.numeric(predict_model)) {
    predict_model <- as.factor(ifelse(predict_model > 0.5, 1, 0))
  }
  predict_model <- as.data.frame(predict_model)
  predict_model$True_label <- test_filtered$Class
  colnames(predict_model) <- c("Predicted_label", "True_label")
  #print(head(predict_model))
  predict <- as.factor(predict_model$Predicted_label)
  actual <- as.factor(predict_model$True_label)

  cm <- confusionMatrix(predict, actual)
  cm_matrix <- as.matrix(cm$table)
  return(cm)
}

# Function to plot AUC-ROC curve
plot_auc_roc <- function(probs, actual_labels, model_name) {
  roc_obj <- roc(actual_labels, probs)
  auc_value <- auc(roc_obj)
  ggroc(roc_obj, legacy.axes = TRUE) +
    ggtitle(paste("AUC-ROC Curve for", model_name, "- AUC:", round(auc_value, 2))) +
    xlab("False Positive Rate") +
    ylab("True Positive Rate") +
    theme_minimal()
}

# Metrics
calculate_metrics <- function(cm) {
  accuracy <- sum(diag(cm)) / sum(cm)
  precision <- cm[2,2] / sum(cm[,2])
  recall <- cm[2,2] / sum(cm[2,])
  f1_score <- 2 * precision * recall / (precision + recall)
  return(c(accuracy, precision, recall, f1_score))
}

# PLot CM
plot_result <- function(cm_table, name) {
  ggplot(data = cm_table, aes(x = Reference, y = Prediction, fill = Freq), mask = TRUE) +
    geom_tile() +
    scale_fill_gradient(low = "white", high = "red") +
    geom_text(aes(label = Freq), color = "black") +
    labs(title = name,
         x = "Actual",
         y = "Predicted") +
    theme_minimal()
}



# Neural Network:
getLayerSize <- function(X, y, hidden_neurons, train=TRUE) {
  n_x <- dim(X)[1]
  n_h <- hidden_neurons
  n_y <- dim(y)[1]

  size <- list("n_x" = n_x,
               "n_h" = n_h,
               "n_y" = n_y)

  return(size)
}

#layer_size <- getLayerSize(X_train, y_train, hidden_neurons = 4)
#layer_size

initializeParameters <- function(X, list_layer_size){

  m <- dim(data.matrix(X))[2]

  n_x <- list_layer_size$n_x
  n_h <- list_layer_size$n_h
  n_y <- list_layer_size$n_y

  W1 <- matrix(runif(n_h * n_x), nrow = n_h, ncol = n_x, byrow = TRUE) * 0.01
  b1 <- matrix(rep(0, n_h), nrow = n_h)
  W2 <- matrix(runif(n_y * n_h), nrow = n_y, ncol = n_h, byrow = TRUE) * 0.01
  b2 <- matrix(rep(0, n_y), nrow = n_y)

  params <- list("W1" = W1,
                 "b1" = b1,
                 "W2" = W2,
                 "b2" = b2)

  return (params)
}
#init_params <- initializeParameters(X_train, layer_size)
#lapply(init_params, function(x) dim(x))


sigmoid <- function(x){
  return(1 / (1 + exp(-x)))
}

forwardPropagation <- function(X, params, list_layer_size){

  m <- dim(X)[2]
  n_h <- list_layer_size$n_h
  n_y <- list_layer_size$n_y

  W1 <- params$W1
  b1 <- params$b1
  W2 <- params$W2
  b2 <- params$b2

  b1_new <- matrix(rep(b1, m), nrow = n_h)
  b2_new <- matrix(rep(b2, m), nrow = n_y)

  Z1 <- W1 %*% X + b1_new
  A1 <- sigmoid(Z1)
  Z2 <- W2 %*% A1 + b2_new
  A2 <- sigmoid(Z2)

  cache <- list("Z1" = Z1,
                "A1" = A1,
                "Z2" = Z2,
                "A2" = A2)

  return (cache)
}

#fwd_prop <- forwardPropagation(X_train, init_params, layer_size)
#lapply(fwd_prop, function(x) dim(x))

computeCost <- function(A2, Y, params, m){

  W1 <- params$W1
  W2 <- params$W2

  logprobs <- Y * log(A2) + (1 - Y) * log(1 - A2)
  cost <- -sum(logprobs) / m

  return(cost)
}

cost <- computeCost(fwd_prop$A2, y_train, init_params, m = dim(X_train)[2])

backwardPropagation <- function(params, cache, X, Y, list_layer_size){

  m <- dim(X)[2]
  n_x <- list_layer_size$n_x
  n_h <- list_layer_size$n_h
  n_y <- list_layer_size$n_y

  W1 <- params$W1
  W2 <- params$W2

  A1 <- cache$A1
  A2 <- cache$A2

  dZ2 <- A2 - Y
  dW2 <- (1 / m) * dZ2 %*% t(A1)
  db2 <- (1 / m) * rowSums(dZ2)

  dZ1 <- t(W2) %*% dZ2 * A1 * (1 - A1)
  dW1 <- (1 / m) * dZ1 %*% t(X)
  db1 <- (1 / m) * rowSums(dZ1)

  grads <- list("dW1" = dW1,
                "db1" = db1,
                "dW2" = dW2,
                "db2" = db2)

  return(grads)
}

#grads <- backwardPropagation(init_params, fwd_prop, X_train, y_train, layer_size)

updateParameters <- function(params, grads, learning_rate){

  W1 <- params$W1
  b1 <- params$b1
  W2 <- params$W2
  b2 <- params$b2

  dW1 <- grads$dW1
  db1 <- grads$db1
  dW2 <- grads$dW2
  db2 <- grads$db2

  W1 <- W1 - learning_rate * dW1
  b1 <- b1 - learning_rate * db1
  W2 <- W2 - learning_rate * dW2
  b2 <- b2 - learning_rate * db2

  params <- list("W1" = W1,
                 "b1" = b1,
                 "W2" = W2,
                 "b2" = b2)

  return(params)
}

#update_params <- updateParameters(init_params, grads, learning_rate = 0.01)
#lapply(update_params, function(x) dim(x))

nn_model <- function(X, Y, n_h, num_iterations = 10000, learning_rate = 0.01){

  list_layer_size <- getLayerSize(X, Y, n_h)
  params <- initializeParameters(X, list_layer_size)

  for (i in 1:num_iterations){

    cache <- forwardPropagation(X, params, list_layer_size)
    cost <- computeCost(cache$A2, Y, params, m = dim(X)[2])
    grads <- backwardPropagation(params, cache, X, Y, list_layer_size)
    params <- updateParameters(params, grads, learning_rate)

    if (i %% 1000 == 0){
      print(paste("Cost after iteration", i, ":", cost))
    }
  }

  return(params)
}

#set.seed(32876688)
#trained_params <- nn_model(X_train, y_train, n_h = 4, num_iterations = 10000, learning_rate = 0.01)

predict_probs <- function(params, X, list_layer_size){
  cache <- forwardPropagation(X, params, list_layer_size)
  return (cache$A2)
}

predict_nn <- function(params, X, list_layer_size){
  cache <- forwardPropagation(X, params, list_layer_size)
  predictions <- ifelse(cache$A2 > 0.5, 1, 0)
  return(predictions)
}

install.packages("devtools")
