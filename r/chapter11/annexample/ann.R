# Practical Machine Learning
# Neural Networks (to predict the strength of concrete)
# Chapter 11

conc <- read.csv("concrete.csv")
str(conc)


normalize <- function(x) {
  return((x - min(x))/(max(x) - min(x)))
}


conc_norm <- as.data.frame(lapply(conc, normalize))

summary(conc_norm$strength)
summary(conc$strength)


conc_train <- conc_norm[1:773, ]
conc_test <- conc_norm[774:1030, ]

library(neuralnet)

# deault: 1 hidden nodes
conc_model <- neuralnet(strength ~ cement + slag + ash + water + superplastic + coarseagg + fineagg + age, data = conc_train)
plot(conc_model)

model_results <- compute(conc_model, conc_test[1:8])

pred_strength <- model_results$net.result

cor(pred_strength, conc_test$strength)


# 5 hidden layers
conc_model2 <- neuralnet(strength ~ cement + slag + ash + water + superplastic + coarseagg + fineagg + age, data = conc_train, hidden = 5)
plot(conc_model2)

model_results2 <- compute(conc_model2, conc_test[1:8])

pred_strength2 <- model_results2$net.result

cor(pred_strength2, conc_test$strength)

