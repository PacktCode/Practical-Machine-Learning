# Practical Machine learning
# Naive Bayes example
# Chapter 9


sms_raw <- read.csv("sms_spam.csv", stringsAsFactors = FALSE)

str(sms_raw)

# convert type variable into a factor
sms_raw$type <- factor(sms_raw$type)

str(sms_raw$type)
table(sms_raw$type)

# text mining package
library(tm)

# corpus is a collection of text docs
sms_corpus <- Corpus(VectorSource(sms_raw$text))

print(sms_corpus)

inspect(sms_corpus[1:3])

# convert data to lower case
corpus_clean <- tm_map(sms_corpus, tolower)

# remove numbers from data
corpus_clean <- tm_map(corpus_clean, removeNumbers)

# remove stop words
corpus_clean <- tm_map(corpus_clean, removeWords, stopwords())

# remove punctuation
corpus_clean <- tm_map(corpus_clean, removePunctuation)

# remove white space
corpus_clean <- tm_map(corpus_clean, stripWhitespace)

# create a Document Term Matrix
sms_dtm <- DocumentTermMatrix(corpus_clean)


sms_raw_train <- sms_raw[1:4169, ]
sms_raw_test <- sms_raw[4170:5559, ]

sms_dtm_train <- sms_dtm[1:4169, ]
sms_dtm_test <- sms_dtm[4170:5559, ]

sms_corpus_train <- corpus_clean[1:4169]
sms_corpus_test <- corpus_clean[4170:5559]

prop.table(table(sms_raw_train$type))
prop.table(table(sms_raw_test$type))


library(wordcloud)

wordcloud(sms_corpus_train, min.freq = 40, random.order = FALSE)

spam <- subset(sms_raw_train, type == "spam")
ham <- subset(sms_raw_train, type == "ham")


wordcloud(spam$text, max.words = 40, scale = c(3, 0.5))
wordcloud(ham$text, max.words = 40, scale = c(3, 0.5))

findFreqTerms(sms_dtm_train, 5)

# build a dictionary
sms_dict <- Dictionary(findFreqTerms(sms_dtm_train, 5))

# limit the training and test sets to only words from the dictionary
sms_train <- DocumentTermMatrix(sms_corpus_train, list(dictionary = sms_dict))
sms_test <- DocumentTermMatrix(sms_corpus_test, list(dictionary = sms_dict))

convert_counts <- function(x) {
  x <- ifelse(x > 0, 1, 0)
  x <- factor(x, levels = c(0, 1), labels = c("No", "Yes"))
  return(x)
}


sms_train <- apply(sms_train, MARGIN = 2, convert_counts)
sms_test <- apply(sms_test, MARGIN = 2, convert_counts)


# need e1071 package for Naive Bayes
library(e1071)


sms_classifier <- naiveBayes(sms_train, sms_raw_train$type)
sms_test_pred <- predict(sms_classifier, sms_test)

library(gmodels)
CrossTable(sms_test_pred, sms_raw_test$type, prop.chisq = FALSE, prop.t = FALSE, dnn = c('predicted', 'actual'))


# improve prediction using Laplace estimator

sms_classifier2 <- naiveBayes(sms_train, sms_raw_train$type, laplace = 1)
sms_test_pred2 <- predict(sms_classifier2, sms_test)
CrossTable(sms_test_pred2, sms_raw_test$type, prop.chisq = FALSE, prop.t = FALSE, prop.r = FALSE, dnn = c('predicted', 'actual'))




