# Practical Machine Learning
# Support Vector Machines (SVM)
# Chapter 6


let <- read.csv("letterdata.csv")
str(let)

let_train <- let[1:16000, ]
let_test <- let[16001:20000, ]



# linear kernel (vanilla)

library(kernlab)
let_classifier <- ksvm(letter ~ ., data = let_train, kernel = "vanilladot")
let_classifier

let_pred <- predict(let_classifier, let_test)

head(let_pred)
table(let_pred, let_test$letter)

agreement <- let_pred == let_test$letter
table(agreement)
prop.table(table(agreement))




# RBF kernel

let_classifier2 <- ksvm(letter ~ ., data = let_train, kernel = "rbfdot")
let_pred2 <- predict(let_classifier2, let_test)

agreement2 <- let_pred == let_test$letter
table(agreement2)
prop.table(table(agreement2))

