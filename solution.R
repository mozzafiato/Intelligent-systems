gc()
rm(list = ls())
options(java.parameters = "- Xmx1024m")
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
qplot(seq(length(termFrequency)),sort(termFrequency), xlab = "index", ylab = "Freq")

#Clustering

#of of documents according to co-occurrence of terms
dtm <- DocumentTermMatrix(train_corpus, control = list(weighting=weightTfIdf))
#rowTotals <- apply(dtm , 1, sum)
#dtm <- dtm[rowTotals> 0, ]
mat <- as.matrix(dtm)
k <- c(2, 4, 8, 16)

#nism ziher ce pravilno rise oz. ce je to zahtevano
for(i in 1:length(k)){
  kmeansResult <- kmeans(mat, k[i])
  tsne.proj <- Rtsne(as.matrix(kmeansResult$cluster), perplexity=10, theta=0.2, dims=2, check_duplicates = F)
  df.tsne <-tsne.proj$Y
  #print(qplot(df.tsne[,1],df.tsne[,2], color = kmeansResult$cluster))
  print(qplot(df.tsne[,1],df.tsne[,2], color = train$label))
}

#POS vector for each document
#vrne out of memmory error :/ 

pos_ann <- Maxent_POS_Tag_Annotator()
word_ann <- Maxent_Word_Token_Annotator()
sent_ann <- Maxent_Sent_Token_Annotator()

posvectors <- list()
for(i in 1:length(train_corpus)){

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
print(tags)
print(length(tags))
posvectors[[i]] <- tags
print(posvectors[[i]])
}

# 3.MODELING
#imbalanced
table(train$label)
table(test$label)

#Preparing data
dtm <- DocumentTermMatrix(corpus, control = list(weighting=weightTfIdf))
matrix <- cbind(as.matrix(dtm), data$label)
training_set <- matrix[1:dim(train)[1],]
names(training_set)[ncol(training_set)] <- "label"
testing_set <- matrix[(dim(train)[1]+1):dim(matrix)[1],-1]

# SVM -> works good on texts
# svm with a radial basis kernel
#dela, sam vrne zelo slabe rezultate (accuracy, recall... tudi primer na vajah vrne podobno idk)
model.svm <- ksvm(label ~ ., training_set , kernel = "rbfdot")
predicted <- predict(model.svm, testing_set, type = "response")
t <- table(test$label, predicted)

# Classification accuracy
sum(diag(t))/sum(t)
# Recall
recall <- t[1,1]/sum(t[1,])
# Precision
precision <- t[1,1]/sum(t[,1])
# F1 score
f1 <- (2*recall*precision)/(precision+recall)

#HyperParameter tuning SVM

#vrne error za target
ksvm_task <- makeClassifTask(data = as.data.frame(training_set), target = "label")
discrete_ps <- makeParamSet(
  makeDiscreteParam("C", values = c(0.01, 0.05, 0.1,0.5)),
  makeDiscreteParam("sigma", values = c(0.005, 0.01, 0.05, 0.1,0.5))
)
print(discrete_ps)

ctrl <- makeTuneControlGrid()
rdesc <- makeResampleDesc("CV", iters = 3L)

res <- tuneParams("classif.ksvm", ksvm_task , rdesc, measures=acc, par.set = discrete_ps, control = ctrl)
print(res)


# Naive bayes -> simple and not too bad

model.nb <- naiveBayes(training_set[,-ncol(training_set)], training_set[,ncol(training_set)])
predicted_nb<- predict(model.nb, testing_set)
t <- table(test$label, predicted_nb)

# Classification accuracy
sum(diag(t))/sum(t)
# Recall
recall <- t[1,1]/sum(t[1,])
# Precision
precision <- t[1,1]/sum(t[,1])
# F1 score
f1 <- (2*recall*precision)/(precision+recall)

# hyperparameter
library(caret)

control <- trainControl(method="repeatedcv", number=10, repeats=3)
model <- train(label~., data=training_set, method="nb", trControl=control, tuneLength=5)

#Logistic regression

control <- trainControl(method="repeatedcv", number=10, repeats=3)
model <- train(label~., data=training_set[,-ncol(training_set)], method = 'regLogistic', trControl=control)

