library(ggplot2)
library(tm)
library(readr)

#Prepare files
conn = file("insults/train.tsv", open="r")
train = readLines(conn)
close(conn)

conn = file("insults/test.tsv", open = "r")
test = readLines(conn)
close(conn)

# 1. Cleaning

#Remove punctuation and stopwords
corpus <- Corpus(VectorSource(train))
corpus <- tm_map(corpus, content_transformer(tolower))
corpus <- tm_map(corpus, removePunctuation)
corpus <- tm_map(corpus, removeWords, stopwords('english'))
corpus <- tm_map(corpus, stripWhitespace)
conn = file("english.stop.txt", open="r")
mystopwords = readLines(conn)
mystopwords
close(conn)
corpus <- tm_map(corpus, removeWords, mystopwords)

#Anonymize proper nouns
s <- as.String(content(corpus))
a1 <- annotate(s, Maxent_Sent_Token_Annotator())
a2 <- annotate(s, Maxent_Word_Token_Annotator(), a1)
a3 <- annotate(s, Maxent_POS_Tag_Annotator(), a2)
a3w <- subset(a3, type == "word")
tags <- sapply(a3w$features, `[[`, "POS")
propernouns <- s[a3w[tags == "NNP"]]
propernouns

