
library(ggplot2)
library(tm)
library(readr)
library(NLP)
library(openNLP)
library(rapport)
library(SnowballC)
library(Rtsne)
library(openNLPmodels.en)
library(useful)
library(kernlab)
library(class)
library(e1071)
library(textclean)
library(mlr)
library(caret)
library(RWeka)

setwd("C:/Users/Melanija/Desktop/Gradiva/5 semestar/IS/2. Seminarska")
#setwd("C:/Users/Jana/Documents/fax/3.letnik/1.semester/IS/2.domaca/")

train <- read.table(file = 'insults/train.tsv', sep = '\t', header = TRUE)
test <- read.table(file = 'insults/test.tsv', sep = '\t', header = TRUE)
data <- rbind(train, test)

# 1.Cleaning
clean_data <- function (data) {
  #Remove unknown symbols
  data$text_a <- sapply(data$text_a, function(x) gsub("x[[:alnum:]][[:digit:]]", "", as.character(x)))
  data$text_a <- sapply(data$text_a, function(x) gsub("\\n", "", as.character(x)))
  data$text_a <- sapply(data$text_a, function(x) gsub("\\r", "", as.character(x)))
  data$text_a <- sapply(data$text_a, function(x) gsub("\\t", "", as.character(x)))
  data$text_a <- sapply(data$text_a, function(x) gsub("\\u[[:digit:]][[:digit:]][[:digit:]][[:alnum:]]", "", as.character(x)))
  
  corpus <- Corpus(VectorSource(data$text_a))
  
  #Remove punctuation and stopwords
  corpus <- tm_map(corpus, removeWords, stopwords('english'))
  conn = file("english.stop.txt", open="r")
  mystopwords = readLines(conn)
  close(conn)
  corpus <- tm_map(corpus, removeWords, mystopwords)
  corpus <- tm_map(corpus, removeNumbers)
  corpus <- tm_map(corpus, removePunctuation)
  corpus <- tm_map(corpus, stripWhitespace)
  
  #Anonymize proper nouns
  sent_ann <- Maxent_Sent_Token_Annotator()
  word_ann <- Maxent_Word_Token_Annotator() 
  person_ann <- Maxent_Entity_Annotator(kind = "person")
  location_ann <- Maxent_Entity_Annotator(kind = "location")
  organization_ann <- Maxent_Entity_Annotator(kind = "organization")
  
  entities <- function(annots, kind)
  {
    k <- sapply(annots$features, `[[`, "kind")
    s[annots[k == kind]]
  }
  
  for (i in 1:length(corpus)){
    s <- as.String(content(corpus[[i]]))
    if(nchar(trimws(s)) == 0) next
    
    ann <- annotate(s, list(sent_ann, word_ann, person_ann, location_ann, organization_ann))
    #print(ann)
    #print(entities(ann, "person"))
    #print(entities(ann, "location"))
    #print(entities(ann, "organization"))
    
    corpus <- tm_map(corpus, removeWords, as.vector(entities(ann, "person")))
    corpus <- tm_map(corpus, removeWords, as.vector(entities(ann, "location")))
    corpus <- tm_map(corpus, removeWords, as.vector(entities(ann, "organization")))
    content(corpus[[i]])
  }
  corpus <- tm_map(corpus, stripWhitespace)
  corpus <- tm_map(corpus, content_transformer(tolower))
  corpus <- tm_map(corpus, stemDocument)
  corpus
}

corpus <- clean_data(data)
train_corpus <- corpus[1:dim(train)[1]]
content(train_corpus[[1]])
test_corpus <- corpus[(dim(train)[1]+1):length(corpus)]
content(test_corpus[[1]])

# 2.Exploration

#Plot the frequency of words
tdm <- TermDocumentMatrix(train_corpus)
termFrequency <- rowSums(as.matrix(tdm))
#termFrequency <- subset(termFrequency, termFrequency >= 10)
v <- sort(rowSums(as.matrix(tdm)),decreasing=TRUE)
d <- data.frame(word = names(v),freq=v)
qplot(seq(length(termFrequency)),sort(termFrequency), xlab = "index", ylab = "Freq")
head(d, 10)
#Clustering

#of of documents according to co-occurrence of terms
dtm <- DocumentTermMatrix(train_corpus, control = list(weighting=weightTfIdf))
mat <- as.matrix(dtm)
tsne.proj <- Rtsne(mat, perplexity=10, theta=0.2, dims=2, check_duplicates = F)
df.tsne <-tsne.proj$Y
k <- c(2, 4, 8, 16)

for(i in 1:length(k)){
  kmeansResult <- kmeans(mat, k[i])
  #visualize the cluster assignments
  print(qplot(df.tsne[,1],df.tsne[,2], color = kmeansResult$cluster))
  
}

#plot document representations according to class labels
print(qplot(df.tsne[,1],df.tsne[,2], color = train$label))

#POS vector for each document
pos_ann <- Maxent_POS_Tag_Annotator()
word_ann <- Maxent_Word_Token_Annotator()
sent_ann <- Maxent_Sent_Token_Annotator()

posvectors <- list()
for(i in 1:length(corpus)){
  
  s <- as.String(content(corpus[[i]]))
  if(nchar(trimws(s)) == 0){
    posvectors[[i]] <- NULL
    next
  }
  a1 <- annotate(s, sent_ann)
  a2 <- annotate(s, word_ann, a1)
  a3 <- annotate(s, pos_ann, a2)
  a3w <- subset(a3, type == "word")
  tags <- sapply(a3w$features, `[[`, "POS")
  posvectors[[i]] <- tags
  print(posvectors[[i]])
}

# 3.MODELING
#imbalanced
table(train$label)
table(test$label)

#Preparing data
dtm <- DocumentTermMatrix(corpus, control = list(weighting=weightTfIdf))
matrix <- cbind(as.matrix(dtm),data$label)
training_set <- matrix[1:dim(train)[1],-ncol(matrix)]
testing_set <- matrix[(dim(train)[1]+1):dim(matrix)[1],-ncol(matrix)]

# SVM -> works good on texts DELAA 
# svm with a radial basis kernel

ksvm_model <- function(training_set, testing_set){
  print("Running model ksvm...")
  model.svm <- ksvm(training_set, make.names(as.factor(train$label)), kernel = "rbfdot")
  predicted <- predict(model.svm, testing_set, type = "response")
  t <- table(make.names(test$label), predicted)
  
  # Classification accuracy
  acc <- sum(diag(t))/sum(t)
  # Recall
  recall <- t[1,1]/sum(t[1,])
  # Precision
  precision <- t[1,1]/sum(t[,1])
  # F1 score
  f1 <- (2*recall*precision)/(precision+recall)
  cat("Accuracy: ",acc, "\n")
  cat("F1 score: ",f1, "\n")
  scores <- c(acc,f1)
  scores
}

scores_ksvm <- ksvm_model(training_set,testing_set)

#HyperParameter tuning SVM

library(mda)
library(modeltools)
train_data <- data.frame(matrix[1:dim(train)[1],])
names(train_data)[ncol(train_data)] <- "label_"
train_data$label_ <- make.names(train_data$label_)

ksvm_task <- makeClassifTask(data = train_data, target = "label_")
discrete_ps <- makeParamSet(
  makeDiscreteParam("C", values = c(0.01, 0.05, 0.1,0.5, 1)),
  makeDiscreteParam("sigma", values = c(0.01, 0.05, 0.1,0.5, 0.6))
)
print(discrete_ps)

ctrl <- makeTuneControlGrid()
rdesc <- makeResampleDesc("CV", iters = 3L)

res <- tuneParams("classif.ksvm", ksvm_task , rdesc, measures=acc, par.set = discrete_ps, control = ctrl)
print(res)
#Tune #Op. pars: C=0.5; sigma=0.1

# Stohastic Gradient Boost

gbm_model <- function(training_set, testing_set){
  print("Running model gbm...")
  man_grid <- expand.grid(n.trees = c(50, 100, 150),
                          interaction.depth = c(1, 3, 4),
                          shrinkage = 0.1,
                          n.minobsinnode = 10)
  
  control <- trainControl(method='cv',
                          number=3, 
                          returnResamp='none', 
                          summaryFunction = twoClassSummary, 
                          classProbs = TRUE)
  
  objModel <- train(training_set, 
                    make.names(train$label), 
                    method='gbm', 
                    trControl=control,
                    verbose = FALSE,
                    tuneGrid = man_grid)

  predictions <- predict(object=objModel,testing_set, type='raw')
  t <- table(make.names(test$label), predictions)
  # Classification accuracy
  acc <- sum(diag(t))/sum(t)
  # Recall
  recall <- t[1,1]/sum(t[1,])
  # Precision
  precision <- t[1,1]/sum(t[,1])
  # F1 score
  f1 <- (2*recall*precision)/(precision+recall)
  cat("Accuracy: ",acc, "\n")
  cat("F1 score: ",f1, "\n")
  scores <- c(acc,f1)
  scores
}

scores_gbm <- gbm_model(training_set,testing_set)

# Use of POS tagging
#concatination: term/POS_tag
pos_concat <- list()
for(i in 1:2){
  words <- strsplit(content(corpus[[i]]), " ")
  pos_concat[[i]] <- paste(words[[1]],posvectors[[i]], sep = "", collapse = " ")
}
pos_corpus <- Corpus(VectorSource(pos_concat))
pos_dtm <- DocumentTermMatrix(corpus, control = list(weighting=weightTfIdf))
pos_matrix <- cbind(as.matrix(pos_dtm),data$label)
training_set_pos <- matrix[1:dim(train)[1],-ncol(matrix)]
testing_set_pos <- matrix[(dim(train)[1]+1):dim(matrix)[1],-ncol(matrix)]

# re-evaluating models with POS
scores_ksvm_pos <- ksvm_model(training_set_pos, testing_set_pos)
scores_gbm_pos <- gbm_model(training_set_pos, testing_set_pos)

# 4.UNDERSTANDING
library(dplyr)
library(CORElearn) 
# feature ranking

# filter method
estReliefF <- attrEval(label_ ~ ., train_data, estimator="InfGain", ReliefIterations=30)

N = 20
bestN <- head(sort(estReliefF, decreasing = TRUE), N)
training_set_bestN <- as.matrix(select(train_data,c(names(bestN))))
testing_set_bestN <- as.matrix(select(data.frame(testing_set),c(names(bestN))))

# re-evaluating models
scores_ksvm_bestN <- ksvm_model(training_set_bestN, testing_set_bestN)
scores_gbm_bestN <- gbm_model(training_set_bestN, testing_set_bestN)

# wrapper model

library(randomForest)

train_data$label_ = replace(train_data$label_, train_data$label_=='X0', 0)
train_data$label_ = replace(train_data$label_, train_data$label_=='X1', 1)
train_data$label_ = factor(train_data$label_)

names(train_data)[names(train_data) == "shadowbeard"] <- "_shadowbeard"
names(train_data)[names(train_data) == "shadowshoss"] <- "_shadowshoss"

library("Boruta")
bor <- Boruta(label_~., data=train_data)

# plot(bor, cex.axis=.7, las=2, xlab="", main="Variable Importance")
# stats <- attStats(bor)
# write.table(stats[order(-stats$maxImp),], "wrapper_stats.txt", append = FALSE, sep = " ",row.names = TRUE, col.names = TRUE)
stats <- read.table("wrapper_stats.txt", sep = " ")
bestWN <- head(stats[order(-stats$maxImp),], N)

filterNames = names(sort(bestN, decreasing = TRUE))
wrapperNames = rownames(bestWN)

# Jaccard
listJac <- c()
for (i in (1:N)){
  fil <- filterNames[1:i]
  wra <- wrapperNames[1:i]
  jac <- length(intersect(fil, wra)) / length(union(fil, wra))
  listJac <- c(listJac, jac)
}
listJac
plot(c(seq(1,N)), listJac, xlab="n", ylab="Jaccard")
