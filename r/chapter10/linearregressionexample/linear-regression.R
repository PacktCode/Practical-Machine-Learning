# Regression analysis of insurance charges data
# Author: Kaushik Balakrishnan, PhD; Email: kaushikb258@gmail.com


ins <- read.csv("insurance.csv", stringsAsFactors = TRUE)

str(ins)
summary(ins$charges)
hist(ins$charges)

table(ins$region)

cor(ins[c("age", "bmi", "children", "charges")])

pairs(ins[c("age", "bmi", "children", "charges")])

library(psych)

pairs.panels(ins[c("age", "bmi", "children", "charges")])


ins_model <- lm(charges ~ age + children + bmi + sex + smoker + region, data = ins)

ins_model <- lm(charges ~ ., data = ins)

summary(ins_model)


ins$age2 <- ins$age^2

ins$bmi30 <- ifelse(ins$bmi >= 30, 1, 0)

ins_model2 <- lm(charges ~ age + age2 + children + bmi + sex + bmi30*smoker + region, data = ins)

summary(ins_model2)




 


