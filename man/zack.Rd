\name{zack}
\alias{zack}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{
%%  ~~function to do ... ~~
}
\description{
%%  ~~ A concise (1-5 lines) description of what the function does. ~~
  This is a R package which implemented a Shallow neural network
  The utility files adress the caculate Accuracy, Precision, F1 Score. Recall, and AUC-ROC metrics
  The other functions: initial parameter, caculate Loss values, update parameters and training


 To train model:

    initializeParameters, getLayerSize -> setting parameters

    predict_nn, predict_probs -> use for prediction

    nn_model -> run train model

    updateParameters, backwardPropagation, forwardPropagation -> Update params

    computeCost -> Caculate the loss of the mode

    sigmoid -> An activation function

  To evaluate:
    calculate_cm: Compute TP, TN, FP, FN

    plot_result: draw heatmap for confusion matrix

    plot_auc_roc: compute and plot AUC-ROC curve

    calculate_metrics: Compute Accuracy, Precision, Recall, F1 score



}

\usage{
   nn_model(X_train, y_train, n_h, num_iterations, learning_rate)
}
%- maybe also 'usage' for other objects documented here.
\arguments{
  \item{X_train}{training set}
  \item{y_train}{label for traning set}
  \item{n_h}{number of hidden units}
  \item{num_iterations}{number of epochs}
  \item{learning_rate}{learning rate value}
}
\details{
%%  ~~ If necessary, more details than the description above ~~
}
\value{
%%  ~Describe the value returned
%%  If it is a LIST, use
%%  \item{comp1 }{Description of 'comp1'}
%%  \item{comp2 }{Description of 'comp2'}
%% ...
}
\references{
%% ~put references to the literature/web site here ~
\url{https://aws.amazon.com/what-is/neural-network}
\url{https://colinfay.me/writing-r-extensions/writing-r-documentation-files.html}
\url{https://rviews.rstudio.com/2020/07/20/shallow-neural-net-from-scratch-using-r-part-1/}
}
\author{
%%  ~~who you are~~
Zack
}
\note{
%%  ~~further notes~~
}

%% ~Make other sections like Warning with \section{Warning }{....} ~

\seealso{
%% ~~objects to See Also as \code{\link{help}}, ~~~
}
%\examples{
%##---- Should be DIRECTLY executable !! ----
%##-- ==>  Define data, use random,
%##--	or do  help(data=index)  for the standard data sets.

%## The function is currently defined as
%function (x)
%{
%  }
%}
% Add one or more standard keywords, see file 'KEYWORDS' in the
% R documentation directory (show via RShowDoc("KEYWORDS")):
% \keyword{ ~kwd1 }
% \keyword{ ~kwd2 }
% Use only one keyword per line.
% For non-standard keywords, use \concept instead of \keyword:
% \concept{ ~cpt1 }
% \concept{ ~cpt2 }
% Use only one concept per line.
