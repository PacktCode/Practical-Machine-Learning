# Practical Machine learning
# Bagging & Random Forest example
# Chapter 13

credit <- read.csv("credit.csv")

library(caret)

m <- train(default ~ ., data = credit, method = "C5.0")
p <- predict(m, credit)

table(p, credit$default)

head(predict(m, credit))
head(predict(m, credit, type = "prob"))

ctrl <- trainControl(method = "cv", number = 10, selectionFunction = "oneSE")

grid <- expand.grid(.model = "tree", .trials = c(1, 5, 10, 15, 20, 25, 30, 35), .winnow = "FALSE")

grid

m <- train(default ~ ., data = credit, method = "C5.0", metric = "Kappa", trControl = ctrl, tuneGrid = grid)
m


library(ipred)

mybag <- bagging(default ~ ., data = credit, nbagg = 25)

credit_pred <- predict(mybag, credit)
table(credit_pred, credit$default)

library(caret)
ctrl <- trainControl(method = "cv", number = 10)
train(default ~ ., data = credit, method = "treebag", trControl = ctrl)



# Bagging

str(svmBag)
svmBag$fit

bagctrl <- bagControl(fit = svmBag$fit, predict = svmBag$pred, aggregate = svmBag$aggregate)

svmBag <- train(default ~ ., data = credit, "bag", trControl = ctrl, bagControl = bagctrl)
svmBag


# Random Forest

library(randomForest)
rf <- randomForest(default ~ ., data = credit)
rf

library(caret)
ctrl <- trainControl(method = "repeatedcv", number = 10, repeats = 10)

grid_rf <- expand.grid(.mtry = c(2, 4, 8, 16))
m_rf <- train(default ~ ., data = credit, method = "rf", metric = "Kappa", trControl = ctrl, tuneGrid = grid_rf)


grid_c50 <- expand.grid(.model = "tree", .trials = c(10, 20, 30, 40), .winnow = "FALSE")
m_c50 <- train(default ~ ., data = credit, method = "C5.0", metric = "Kappa", trControl = ctrl, tuneGrid = grid_c50)

m_rf
m-c50


