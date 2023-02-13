rm(list=ls())
cat("\014")
# set working directory
setwd("/Users/tanmaybirar/Documents/semantics")  

# install.packages("tm")
# install.packages("e1071")
# install.packages("SnowballC")
library(tm)  # Text Mining Package
library(e1071)  # Functions for support vector machines, etc.
library(SnowballC)  # An R interface that implements Porter's word stemming algorithm for collapsing words to a common root 
library(caret)
library(tm)
library(lsa)
library(topicmodels)
library(wordcloud)

#STEP 1 - LOAD THE DATASET
tweets <- read.csv("tweet.csv")
head(tweets)
#Plot the frequency of tweets:
library(ggplot2)
attach(tweets)
ggplot(tweets, aes(label), x_lab = "Class", y_lab = "Frequency") +
  geom_bar(fill = "gray50")

#STEP 2 Construct corpus, clean text

corp <- Corpus(VectorSource(tweets$tweet))


# text cleaning
corp <- tm_map(corp, removePunctuation) 
dtm <- DocumentTermMatrix(corp)
as.matrix(dtm) # show all the terms

corp <- tm_map(corp, stripWhitespace) 
dtm <- DocumentTermMatrix(corp)
as.matrix(dtm) # show all the terms

corp <- tm_map(corp, removeWords, stopwords("english"))
dtm <- DocumentTermMatrix(corp)
as.matrix(dtm) # show all the terms

corp <- tm_map(corp, stemDocument)  # Stem words to get the word root
dtm <- DocumentTermMatrix(corp)
as.matrix(dtm) # show all the terms

corp <- tm_map(corp, removeNumbers)  
dtm <- DocumentTermMatrix(corp)
as.matrix(dtm) # show all the terms

# Term Frequency – Inverse Document Frequency (TF-IDF) 
tfidf <- weightTfIdf(dtm)
inspect(tfidf)

# interprets each row of tweet as a document
mycorpus <- Corpus(VectorSource(tweets$tweet))

# define customized stopwords
mystop <- c("flight", "flights", "fly", "travel", "traveller", "amp", "&amp","@","#","httptcojqrcqzw")  
# define the text cleaning process
dtm.control = list(tolower=T, removePunctuation=T, removeNumbers=T, stopwords=c(stopwords("english"), mystop), 
                   stripWhitespace=T, stemming=T)
# generate document-term matrix
dtm.full <- DocumentTermMatrix(mycorpus, control=dtm.control)  
inspect(dtm.full)

#STEP 3:
# remove terms occurring in less than 1% of the documents
dtm <- removeSparseTerms(dtm.full, 0.99)  
dim(dtm.full)  
dim(dtm)  


#STEP 4
# Estimate the LDA model 
# LDA(x, k): x is a document term matrix, k is the # of topics
lda.model = LDA(dtm[1:100,], 5) # use the first 100 documents

# top terms of each topic
# terms(x, k): x is the topic model, k is the maximum # of terms returned
freqterms <- terms(lda.model,30)
freqterms
wordcloud(freqterms, max.words = 30, colors = brewer.pal(6,"Dark2"))

# get the posterior probabilities of the model
myposterior <- posterior(lda.model) 
# TOPIC distribution in each DOCUMENT, each row is a document, each column is a topic
topics = myposterior$topics 
# TERM distribution in each TOPIC, each row is a topic, each column is to a term
terms = myposterior$terms

# Plot word cloud for a specific topic
set.seed(123)
tid <- 3 # topic id, it can be 1, ... k you specified in LDA(dtm, k)
freq <- terms[tid, ] # the probability of each term in a given topic
wordcloud(names(freq), freq, max.words=30, colors=brewer.pal(6,"Dark2"))

set.seed(123)
tid <- 5 # topic id, it can be 1, ... k you specified in LDA(dtm, k)
freq <- terms[tid, ] # the probability of each term in a given topic
wordcloud(names(freq), freq, max.words=30, colors=brewer.pal(6,"Dark2"))

set.seed(123)
tid <- 1 # topic id, it can be 1, ... k you specified in LDA(dtm, k)
freq <- terms[tid, ] # the probability of each term in a given topic
wordcloud(names(freq), freq, max.words=30, colors=brewer.pal(6,"Dark2"))

set.seed(123)
tid <- 2 # topic id, it can be 1, ... k you specified in LDA(dtm, k)
freq <- terms[tid, ] # the probability of each term in a given topic
wordcloud(names(freq), freq, max.words=30, colors=brewer.pal(6,"Dark2"))

set.seed(123)
tid <- 3 # topic id, it can be 1, ... k you specified in LDA(dtm, k)
freq <- terms[tid, ] # the probability of each term in a given topic
wordcloud(names(freq), freq, max.words=30, colors=brewer.pal(6,"Dark2"))

set.seed(123)
tid <- 4 # topic id, it can be 1, ... k you specified in LDA(dtm, k)
freq <- terms[tid, ] # the probability of each term in a given topic
wordcloud(names(freq), freq, max.words=30, colors=brewer.pal(6,"Dark2"))

set.seed(123)
tid <- 5 # topic id, it can be 1, ... k you specified in LDA(dtm, k)
freq <- terms[tid, ] # the probability of each term in a given topic
wordcloud(names(freq), freq, max.words=30, colors=brewer.pal(6,"Dark2"))


#STEP 7
# get the input variables X and output Y
X <- as.data.frame(as.matrix(dtm))
Y <- as.factor(tweets$label)

# split the data into training and test data sets
set.seed(123)   # for reproducible results
#train <- sample(1:nrow(dtm),(2/3)*nrow(dtm))
train = sample(1:nrow(dtm),(2/3)*nrow(dtm))
test = sample(1:nrow(dtm), (1/3)*nrow(dtm))

###########################################
##########  K-Nearest Neighbors  ##########
###########################################
# 10-fold cross-validation
ctrl <- trainControl(method="cv", number=10) 
knnFit <- train(X[train,], Y[train], method = "knn", trControl = ctrl, tuneGrid = expand.grid(k = 1:10))
# plot the # of neighbors vs. accuracy (based on repeated cross validation)
plot(knnFit)

A# Evaluate classifier performance on testing data
actual <- Y[test]
pred <- predict(knnFit, X[test,])
cm1 <- confusionMatrix(pred, actual, positive="1")
cm1


#STEP 9 : Support Vector Machine

# set kernel='linear'
svm.model <- svm(X[train, ], Y[train], kernel='linear')
pred <- predict(svm.model, X[test,])
# get confusion matrix
cm2 <- confusionMatrix(pred, Y[test], positive="1")
cm2

#Compare KNN and SVM

result = rbind(cm1$byClass[c("Sensitivity", "Specificity", "Balanced Accuracy")], 
               cm2$byClass[c("Sensitivity", "Specificity", "Balanced Accuracy")])

row.names(result) <- c("KNN", "SVM")
result
