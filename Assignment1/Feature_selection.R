library(caret)
library(lattice)
library(ranger)

genes <- read.csv("DLBCL.csv", header = T, sep = ",")

#removing the id attribute
genes[1] <- NULL

inTraining <- createDataPartition(genes$class, p = 0.70, list = FALSE)
training <- genes[ inTraining,]
testing  <- genes[-inTraining,]

F1score <- function(observed, predicted, pos.class)
{
  t <- table(observed, predicted)
  neg.class <- which(row.names(t) != pos.class)
  (t[pos.class, pos.class] * 2) / ( 2* t[pos.class, pos.class] + t[neg.class, pos.class] +  t[pos.class, neg.class])
}

myMonitor <- function(obj){
  i <- which.max(obj@fitness)
  points(obj@iter, sum(obj@population[i,]), col = "red")
}

myFitness <- function(vector){
  
  n <- sum(vector)
  if (n >= 2 && n <= 1000){
    vector[length(vector)] = 1
    
    chosen <- training[,!!vector]
    
    train_control<- trainControl(method="cv", number=3, savePredictions = TRUE)
    
    model <- train(class~., data=chosen, trControl=train_control, method="ranger")
    
    matrix <- confusionMatrix(model)
    accuracy <- sum(diag(matrix$table))/100
    accuracy/n
    
  } else {
    0
  }
}

iter = 30
plot("iteration","selected features", xlim = c(1,iter), ylim=c(300,600))
GA <- ga(type = "binary", fitness = myFitness, nBits = ncol(genes), parallel = TRUE, maxiter = iter, popSize = 15, monitor = myMonitor)

#Final model
vector = GA@solution[1,]
vector[length(vector)] = 1
chosen <- training[,!!vector]

train_control<- trainControl(method="cv", number=3, savePredictions = TRUE)
model <- train(class~., data=chosen, trControl=train_control, method="ranger")

predictions <- predict(model,testing)

#F1 score
t <- table(testing$class, predictions)
f1 <- F1score(testing$class, predictions, "DLBCL")
f1

#Accuracy
matrix <- confusionMatrix(model)
accuracy <- sum(diag(matrix$table))/length(testing$class)
accuracy

#Number of selected features:
NoFeatures <- sum(GA@solution[1,])

#List of the selected features:
columns <- genes[1,] 
selected_features=columns[GA@solution[1,]==1]

#Indexes of selected features:
ind = c(1:ncol(genes))
selected_features_index <- ind[GA@solution[1,]==1]

#correlation of columns with the column class
class <- rep(0, length(genes$class))
class[genes$class == 'DLBCL'] <- 1
correlation <- cor(x=genes[,1:length(genes)-1], y=class)
correlation <- abs(correlation)

#get the first n (=NoFeatures) indexes of columns with maximal correlation
corr_features <- which(-correlation<=sort(-correlation)[NoFeatures], arr.ind = TRUE)

#intersection between columns of fitness function and correlated columns
c <- intersect(selected_features_index, corr_features)
#percentage of mutual features
length(c)/NoFeatures





