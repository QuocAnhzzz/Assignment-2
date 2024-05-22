# Assignment-2

  This is a R package which implemented a Shallow neural network
  The utility files adress the caculate Accuracy, Precision, F1 Score. Recall, and AUC-ROC metrics
  The other functions: initial parameter, caculate Loss values, update parameters and training


 ## To train model:

    initializeParameters, getLayerSize -> setting parameters

    predict_nn, predict_probs -> use for prediction

    nn_model -> run train model

    updateParameters, backwardPropagation, forwardPropagation -> Update params

    computeCost -> Caculate the loss of the mode

    sigmoid -> An activation function

  ## To evaluate:
    calculate_cm: Compute TP, TN, FP, FN

    plot_result: draw heatmap for confusion matrix

    plot_auc_roc: compute and plot AUC-ROC curve

    calculate_metrics: Compute Accuracy, Precision, Recall, F1 score

# Reference
  James et al. (2021): https://www.casact.org/sites/default/files/2022-12/James-G.-et-al.-2nd-edition-Springer-2021.pdf
