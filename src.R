# Configuration of environment
rm(list=ls())
clc <- function() cat("\014 Screen has been succesfully wiped")
dyn.load('/Library/Java/JavaVirtualMachines/jdk-10.jdk/Contents/Home/lib/server/libjvm.dylib')

Sys.setenv(LANG = "en")
library(RWeka)
library(partykit)
library(caret)
library(MLmetrics)
library(reshape2)

clc()

########____OBLICZANIE_JAKOŚCI_KLASYFIKACJI____########

class_classification_quality <- function(x, confusion_matrix)
{
  ###___obliczenie_zmiennych___###
  true_positive = confusion_matrix[x, x]
  
  false_positive = 0
  for(i in 1:ncol(confusion_matrix))
  {
    if(i == x) {}
    else
    {
      false_positive = false_positive + confusion_matrix[i, x]
    }
    
  }
  
  false_negative = 0
  for(i in 1:ncol(confusion_matrix))
  {
    if(i == x) {}
    else
    {
      false_negative = false_negative + confusion_matrix[x, i]
    }
  
  }
  
  ###___definiowanie_parametrów___###
  precision <- function(true_positive, false_positive)
  {
    if (false_positive == 0)
    {
      if (true_positive == 0)
      {
        return(0)
      } else {
        return(1)
      }
    } else {
      return(true_positive / (true_positive + false_positive))
    }
  }
  
  
  recall <- function(true_positive, false_negative)
  {
    if (true_positive == 0 & false_negative == 0)
    {
      return(0)
    } else {
      return(true_positive / (true_positive + false_negative))
    }
  }
  
  
  fscore <- function(true_positive, false_positive, false_negative)
  {
    prec <- precision(true_positive, false_positive)
    rec <- recall(true_positive, false_negative)
    if (prec + rec == 0)
    {
      return(0)
    } else {
      return(2 * prec * rec / (prec + rec))
    }
  }
  
  ###___definiowanie_parametrów___###
  val_1 = precision(true_positive, false_positive)
  val_2 = recall(true_positive, false_negative)
  val_3 = fscore(true_positive, false_positive, false_negative)
  return(c(val_1, val_2, val_3))
}

quality_of_classification <- function(confusion_matrix)
{
  mean_precision = 0
  mean_recall = 0
  mean_fscore = 0
  normalise = 1 / nrow(confusion_matrix)
  for(x in 1:nrow(confusion_matrix))
  {
    a <- class_classification_quality(x, confusion_matrix)
    mean_precision = mean_precision + a[1]
    mean_recall = mean_recall + a[2]
    mean_fscore = mean_fscore + a[3]
  }
  mean_precision = mean_precision * normalise
  mean_recall = mean_recall * normalise
  mean_fscore = mean_fscore * normalise
  return(c(mean_precision, mean_recall, mean_fscore))
}

clc()

### Main code

my_data <- read.csv(file="wine.data.csv")


###___STRATIFIED_CV___###
tree <- J48(class~., data = my_data) #control = Weka_control(U=TRUE, M=1))
precision <- c()
recall <- c()
fscore <- c()
acc <- c()
amount_of_folds <- c(2:60)

for(i in amount_of_folds)
{
  evaluate_tree <- evaluate_Weka_classifier(tree,
                                            numFolds = i, complexity = TRUE,
                                            seed = 123, class = TRUE)
  temp <- quality_of_classification(evaluate_tree$confusionMatrix)
  
  precision <- rbind(precision, temp[1])
  recall <- rbind(recall, temp[2])
  fscore <- rbind(fscore, temp[3])
  acc <- rbind(acc, (evaluate_tree$details)[["pctCorrect"]])
}

#___PLOTTING_A_FIGURE___#
par(pch=22, col="blue")
par(mfrow=c(2,2))
plot(amount_of_folds, precision, type="l", main="precision: ", xlab="amount of folds", ylab="score", col = "red")
plot(amount_of_folds, acc, type="l", main="accuracy:", xlab="amount of folds", ylab="score", col = "red")
plot(amount_of_folds, recall, type="l", main="recall:", xlab="amount of folds", ylab="score", col = "red")
plot(amount_of_folds, fscore, type="l", main="fscore:", xlab="amount of folds", ylab="score", col = "red")


###___NONSTRATIFIED_CV___###
precision <- c()
recall <- c()
fscore <- c()
acc <- c()
for(AF in amount_of_folds)
{
  nfolds= AF
  t_acc=0
  t_prec=0
  t_rec=0
  t_fscore=0
  mydata <- my_data[sample(nrow(my_data)),]
  folds <-cut(seq(1, nrow(mydata)), breaks = nfolds, labels = FALSE)
  for (i in 1:nfolds)
    {
      testIndices <- which(folds == i, arr.ind = TRUE)
      testData <- mydata[testIndices, ]
      trainData <- mydata[-testIndices, ]
      little_tree <- J48(class~., data=trainData)
      evaluate_little_tree <- evaluate_Weka_classifier(little_tree, 
                                     newdata = testData)
      t_acc = t_acc + (evaluate_little_tree$details)[["pctCorrect"]]
      t_prec = t_prec + quality_of_classification(evaluate_little_tree$confusionMatrix)[1]
      t_rec = t_rec + quality_of_classification(evaluate_little_tree$confusionMatrix)[2]
      t_fscore = t_fscore + quality_of_classification(evaluate_little_tree$confusionMatrix)[3]
    }
  
  precision <- rbind(precision, t_prec/nfolds)
  recall <- rbind(recall, t_rec/nfolds)
  fscore <- rbind(fscore, t_fscore/nfolds)
  acc <- rbind(acc, t_acc/nfolds)
  
}

#___PLOTTING_A_FIGURE___#
par(pch=22, col="blue")
par(mfrow=c(2,2))
plot(amount_of_folds, precision, type="l", main="precision: ", xlab="amount of folds", ylab="score", col = "red")
plot(amount_of_folds, acc, type="l", main="accuracy:", xlab="amount of folds", ylab="score", col = "red")
plot(amount_of_folds, recall, type="l", main="recall:", xlab="amount of folds", ylab="score", col = "red")
plot(amount_of_folds, fscore, type="l", main="fscore:", xlab="amount of folds", ylab="score", col = "red")



#___TESTY_PRARMETRÓW_DRZEWA___U_O#
wine <- read.csv(file="wine.data.csv")
glass <- read.csv(file="glass.data.csv")
diabetes <- read.csv(file="diabetes.data.csv")
param_tree <- J48(class~., data = diabetes, control = Weka_control(U=FALSE, O=FALSE, C= 0,1,0.05 ))
evaluate_param_tree <- evaluate_Weka_classifier(param_tree,
                                          numFolds = 50, complexity = TRUE,
                                          seed = 123, class = TRUE)

quality_of_classification(evaluate_param_tree$confusionMatrix)


#___TESTY_PRARMETRÓW_DRZEWA___C#
par(pch=22, col="blue")
par(mfrow=c(2,2))

amount <- seq(0.05, 0.5, by=0.05)
fscore=c()
for(i in seq(0.05, 0.5, by=0.05)){
  glass <- read.csv(file="glass.data.csv")
  param_tree <- J48(class~., data=glass, control=Weka_control(U=FALSE, O=TRUE ,C=i))
  eval_param_tree <- evaluate_Weka_classifier(param_tree,
                                numFolds = 10, complexity = TRUE,
                                seed = 123, class = TRUE, U=TRUE)
  
  fscore <- rbind(fscore, quality_of_classification(eval_param_tree$confusionMatrix)[3])
}

plot(amount, fscore, type="l", main="Glass:",  xlab="value of C", ylab="fscore", col = "red")

fscore=c()
for(i in seq(0.05, 0.5, by=0.05)){
  diabetes <- read.csv(file="diabetes.data.csv")
  param_tree <- J48(class~., data=diabetes, control=Weka_control(U=FALSE, O=TRUE ,C=i))
  eval_param_tree <- evaluate_Weka_classifier(param_tree,
                                              numFolds = 10, complexity = TRUE,
                                              seed = 123, class = TRUE, U=TRUE)
  
  fscore <- rbind(fscore, quality_of_classification(eval_param_tree$confusionMatrix)[3])
}

plot(amount, fscore, type="l", main="Diabetes:", xlab="value of C", ylab="fscore", col = "red")

fscore=c()
for(i in seq(0.05, 0.5, by=0.05)){
  wine <- read.csv(file="wine.data.csv")
  param_tree <- J48(class~., data=wine, control=Weka_control(U=FALSE, O=TRUE ,C=i))
  eval_param_tree <- evaluate_Weka_classifier(param_tree,
                                              numFolds = 10, complexity = TRUE,
                                              seed = 123, class = TRUE, U=TRUE)
  
  fscore <- rbind(fscore, quality_of_classification(eval_param_tree$confusionMatrix)[3])
}

plot(amount, fscore, type="l", main="Wines:", xlab="value of C", ylab="fscore", col = "red")


#___TESTY_PRARMETRÓW_DRZEWA___M#
par(pch=22, col="blue")
par(mfrow=c(2,2))

amount <- seq(2, 20)
fscore=c()
for(i in seq(2, 20)){
  glass <- read.csv(file="glass.data.csv")
  param_tree <- J48(class~., data=glass, control=Weka_control(U=FALSE, O=TRUE, M=i))
  eval_param_tree <- evaluate_Weka_classifier(param_tree,
                                              numFolds = 10, complexity = TRUE,
                                              seed = 123, class = TRUE, U=TRUE)
  
  fscore <- rbind(fscore, quality_of_classification(eval_param_tree$confusionMatrix)[3])
}

plot(amount, fscore, type="l", main="Glass:",  xlab="value of C", ylab="fscore", col = "red")

fscore=c()
for(i in seq(2, 20)){
  diabetes <- read.csv(file="diabetes.data.csv")
  param_tree <- J48(class~., data=diabetes, control=Weka_control(U=FALSE, O=TRUE, M=i))
  eval_param_tree <- evaluate_Weka_classifier(param_tree,
                                              numFolds = 10, complexity = TRUE,
                                              seed = 123, class = TRUE, U=TRUE)
  
  fscore <- rbind(fscore, quality_of_classification(eval_param_tree$confusionMatrix)[3])
}

plot(amount, fscore, type="l", main="Diabetes:", xlab="value of C", ylab="fscore", col = "red")

fscore=c()
for(i in seq(2, 20)){
  wine <- read.csv(file="wine.data.csv")
  param_tree <- J48(class~., data=wine, control=Weka_control(U=FALSE, O=TRUE,R=FALSE,M=i))
  eval_param_tree <- evaluate_Weka_classifier(param_tree,
                                              numFolds = 10, complexity = TRUE,
                                              seed = 123, class = TRUE, U=TRUE)
  
  fscore <- rbind(fscore, quality_of_classification(eval_param_tree$confusionMatrix)[3])
}

plot(amount, fscore, type="l", main="Wines:", xlab="value of C", ylab="fscore", col = "red")



#___TESTY_PRARMETRÓW_DRZEWA___R#
wine <- read.csv(file="wine.data.csv")
glass <- read.csv(file="glass.data.csv")
diabetes <- read.csv(file="diabetes.data.csv")
param_tree <- J48(class~., data = glass, control = Weka_control(U=FALSE, O=FALSE, M=5, R=FALSE ))
evaluate_param_tree <- evaluate_Weka_classifier(param_tree,
                                                numFolds = 50, complexity = TRUE,
                                                seed = 123, class = TRUE)

quality_of_classification(evaluate_param_tree$confusionMatrix)




#___NAJLEPSZE DRZEWO___#
param_tree <- J48(class~., data = glass, control = Weka_control(C=0.3))
evaluate_param_tree <- evaluate_Weka_classifier(param_tree,
                                                numFolds = 50, complexity = TRUE,
                                                seed = 123, class = TRUE)

quality_of_classification(evaluate_param_tree$confusionMatrix)
plot(param_tree)
heatmap(evaluate_param_tree$confusionMatrix)













