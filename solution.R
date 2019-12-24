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
library(factoextra)

 setwd("C:/Users/Melanija/Desktop/Gradiva/5 semestar/IS/2. Seminarska")
#setwd("C:/Users/Jana/Documents/fax/3.letnik/1.semester/IS/2.domaca/")

train <- read.table(file = 'insults/train.tsv', sep = '\t', header = TRUE)
test <- read.table(file = 'insults/test.tsv', sep = '\t', header = TRUE)

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

train_corpus <- clean_data(train)
content(train_corpus[[1]])

# 2.Exploration

#Plot the frequency of words
tdm <- TermDocumentMatrix(corpus)
termFrequency <- rowSums(as.matrix(tdm))
#termFrequency <- subset(termFrequency, termFrequency >= 10)
qplot(seq(length(termFrequency)),sort(termFrequency), xlab = "index", ylab = "Freq")

#Clustering

#of of documents according to co-occurrence of terms
dtm <- DocumentTermMatrix(corpus, control = list(weighting=weightTfIdf))
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
posvectors <- c()
for(i in 1:length(corpus)){
posvectors[[i]] <- c()
s <- as.String(content(corpus[[i]]))
a1 <- annotate(s, Maxent_Sent_Token_Annotator())
a2 <- annotate(s, Maxent_Word_Token_Annotator(), a1)
a3 <- annotate(s, Maxent_POS_Tag_Annotator(), a2)
a3w <- subset(a3, type == "word")
tags <- sapply(a3w$features, `[[`, "POS")
print(tags)
print(length(tags))
posvectors[[i]] <- tags
}

# 3.MODELING
#imbalanced
table(train$label)
table(test$label)

# Identify the class column
train_corpus <- clean_data(test)
content(test_corpus[[1]])
test_tdm <- TermDocumentMatrix(test_corpus)

#
#
# NE VEM CE JE PRAVILNO?
#
#

# SVM -> works good on texts
# svm with a radial basis kernel
model.svm <- ksvm(label ~ ., tdm, kernel = "rbfdot")
predicted <- predict(model.svm, test_tdm, type = "response")
t <- table(observed, predicted)

# Classification accuracy
sum(diag(t))/sum(t)

# Recall
recall <- t[1,1]/sum(t[1,])

# Precision
precision <- t[1,1]/sum(t[,1])

# F1 score
f1 <- (2*recall*precision)/(precision+recall)

# Naive bayes -> simple and not too bad

#
#
# nevem kako na isto dolzino....
#
#

model.nb <- naiveBayes(as.matrix(tdm), as.factor(train$label))
rs<- predict(model, as.matrix((test$label)))

# hyperparameter
library(caret)

control <- trainControl(method="repeatedcv", number=10, repeats=3)
model <- train(label~., data=train, method="nb", trControl=control, tuneLength=5)
