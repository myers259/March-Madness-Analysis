library(tidyverse)
library(car)
library(caret)
library(glmnet)

cbb <- cbbUpdated %>% select(-X)
row.names(cbb) <- paste(cbb$TEAM,cbb$YEAR, sep = "_")
cbb <- cbb %>% select(-c("TEAM", "YEAR", "RegGames"))

cbb$Power5 <- ifelse(cbb$CONF %in% c("ACC", "B10", "B12", "SEC", "P12"), 1, 0)

cbb <- cbb %>% select(-"CONF")

cbb$SEED <- replace_na(cbb$SEED, "NIT")
cbb$SEED <- as.factor(cbb$SEED)


cbbDummies <- dummyVars(~ ., data=cbb)
c <- as.data.frame(predict(cbbDummies, newdata = cbb))
summary(c)


set.seed(489)
s <- sample(nrow(c), nrow(c)*.8)
train <- c[s,]
test <- c[-s,]


# All subsets ------------------------------------------------------------------
s1 <- sample(nrow(cbb), nrow(cbb)*.8)
trainNonDummy <- cbb[s1,]
testNonDummy <- cbb[-s1,]


library(leaps)
allSub <- regsubsets(POSTSEASON ~ . , data=trainNonDummy, nbest=5)
plot(allSub, scale="adjr2")
bestAllSub <- lm(POSTSEASON ~ ADJOE + ADJDE + SEED + Power5 + DRB, data = trainNonDummy)
summary(bestAllSub)
vif(bestAllSub)
plot(bestAllSub)

rmseAlltr <- sqrt(mean((bestAllSub$residuals)^2))
rmseAlltr

allPred <- predict(bestAllSub, testNonDummy)
rmseAlltest <- sqrt(mean((testNonDummy$POSTSEASON - allPred)^2))
rmseAlltest
rmseAlltest/rmseAlltr
lm.beta(bestAllSub)
# Setup for regularized regression ---------------------------------------------
library(glmnet)
xTrain <- as.matrix(train[,-20]) 
yTrain <- as.matrix(train[,20])

xTest <- as.matrix(test[,-20])
yTest <- as.matrix(test[,20])
# Ridge ------------------------------------------------------------------------
cbbRidge <- cv.glmnet(xTrain, yTrain, alpha=0)
plot(cbbRidge)
cbbRidge$glmnet.fit
lambda1se <- cbbRidge$lambda.1se
ridgePredTest <- predict(cbbRidge, xTest, s="lambda.1se")
rmseRidgeTest<- sqrt(mean((ridgePredTest - yTest)^2))
rmseRidgeTest

ridgePredTrain <- predict(cbbRidge, xTrain, s="lambda.1se")
rmseRidgeTrain<- sqrt(mean((ridgePredTrain - yTrain)^2))
rmseRidgeTrain

rsq <- glmnet(xTrain, yTrain, alpha=0, lambda= lambda1se)
rsq$dev.ratio

rmseRidgeTest/rmseRidgeTrain

# Lasso ------------------------------------------------------------------------
cbbLasso <- cv.glmnet(xTrain, yTrain, alpha=1)
plot(cbbLasso)
lambda1seR <- cbbLasso$lambda.1se
lassoPredTest <- predict(cbbLasso, xTest, s="lambda.1se")
rmseLassoTest <- sqrt(mean((lassoPredTest - yTest)^2))

lassoPredTrain <- predict(cbbLasso, xTrain, s="lambda.1se")
rmseLassoTrain <- sqrt(mean((lassoPredTrain - yTrain)^2))

rsqL <- glmnet(xTrain, yTrain, alpha = 1, lambda = lambda1seR)
rsqL
coef(cbbLasso)
rmseLassoTest/rmseLassoTrain

# Relaxed Lasso ----------------------------------------------------------------
cbbLassoR <- cv.glmnet(xTrain, yTrain, alpha=1, relax = T)
plot(cbbLassoR)
lambda1seR <- cbbLassoR$lambda.1se
lassoPredTestR <- predict(cbbLassoR, xTest, s="lambda.1se")
rmseLassoTestR <- sqrt(mean((lassoPredTestR - yTest)^2))
rmseLassoTestR
lassoPredTrainR <- predict(cbbLassoR, xTrain, s="lambda.1se")
rmseLassoTrainR <- sqrt(mean((lassoPredTrainR - yTrain)^2))
rmseLassoTrainR
rmseLassoTestR/rmseLassoTrainR
rsqLR <- glmnet(xTrain, yTrain, alpha = 1, lambda = lambda1seR, relax=T)
rsqLR$dev.ratio


# Lasso into Ridge -------------------------------------------------------------
xTrainL <- as.matrix(train[,c(3, 4, 19, 21, 29, 30, 31, 37, 38)]) 
yTrainL <- as.matrix(train[,20])

xTestL <- as.matrix(test[,c(3, 4, 19, 21, 29, 30, 31, 37, 38)])
yTestL <- as.matrix(test[,20])


cbbRidgeL <- cv.glmnet(xTrainL, yTrainL, alpha=0)
lambda1seL <- cbbRidgeL$lambda.1se
ridgePredTestL <- predict(cbbRidgeL, xTestL, s="lambda.1se")
rmseRidgeTestL<- sqrt(mean((ridgePredTestL - yTestL)^2))
rmseRidgeTestL

ridgePredTrainL <- predict(cbbRidgeL, xTrainL, s="lambda.1se")
rmseRidgeTrainL<- sqrt(mean((ridgePredTrainL - yTrainL)^2))
rmseRidgeTrainL

rmseRidgeTestL/rmseRidgeTrainL

rsqLr <- glmnet(xTrainL, yTrainL, alpha=0, lambda= lambda1se)
rsqLr$dev.ratio

coef(cbbRidgeL)


summary(cbbRidge)
# Elastic Net ------------------------------------------------------------------
elastic <- cv.glmnet(xTrain, yTrain, alpha = .25)
trainPE <- predict(elastic, xTrain, s="lambda.1se")
rmseTrainE <- sqrt(mean((yTrain - trainPE)^2))
rmseTrainE
testPE <- predict(elastic, xTest, s="lambda.1se")
rmseTestE <- sqrt(mean((yTest - testPE)^2))
rmseTestE
rmseTestE/rmseTrainE
rsqEL <- glmnet(xTrain, yTrain, alpha = .25, lambda = elastic$lambda.1se)
rsqEL$dev.ratio

elastic <- cv.glmnet(xTrain, yTrain, alpha = .5)
trainPE <- predict(elastic, xTrain, s="lambda.1se")
rmseTrainE <- sqrt(mean((yTrain - trainPE)^2))
rmseTrainE
testPE <- predict(elastic, xTest, s="lambda.1se")
rmseTestE <- sqrt(mean((yTest - testPE)^2))
rmseTestE

elastic <- cv.glmnet(xTrain, yTrain, alpha = .75)
trainPE <- predict(elastic, xTrain, s="lambda.1se")
rmseTrainE <- sqrt(mean((yTrain - trainPE)^2))
rmseTrainE
testPE <- predict(elastic, xTest, s="lambda.1se")
rmseTestE <- sqrt(mean((yTest - testPE)^2))
rmseTestE

coef(elastic)

# PCA --------------------------------------------------------------------------
library(car)
library(lm.beta)
library(corrplot)
library(psych)
corr.test(c, adjust = "none")
pca1 <- prcomp(c[-c(20)], scale = T)
plot(pca1)
summary(pca1)
screeplot(pca1)


pcData <- as.data.frame(pca1$x)
pcData$POSTSEASON <- c$POSTSEASON
print(pca1$rotation, cutoff = .4)

round(pca1$rotation[,25], 2)[which(abs(pca1$rotation[,25]) > .25)]
round(pca1$rotation[,33], 2)[which(abs(pca1$rotation[,33]) > .25)]
round(pca1$rotation[,20], 2)[which(abs(pca1$rotation[,20]) > .25)]

s2 <- sample(nrow(pcData), nrow(pcData)*.8)
trainPC <- pcData[s2,]
testPC <- pcData[-s2,]

pcLm <- lm(POSTSEASON ~ ., data = trainPC)
summary(pcLm)
pcLm <- lm(POSTSEASON ~ PC1 + PC3 + PC4 + PC7 + PC9 + PC13 + PC16 + PC18 +  PC20 + PC21 + PC23 + PC25 + PC26 + PC28 + PC33, data = pcData)
summary(pcLm)

rmsePCtr <- sqrt(mean((pcLm$residuals)^2))
rmsePCtr

pcPred <- predict(pcLm, testPC)
rmsePCtest <- sqrt(mean((testPC$POSTSEASON - pcPred)^2))
rmsePCtest
rmsePCtest/rmsePCtr

stdCoef <- lm.beta(pcLm)
sort(abs(stdCoef$coefficients))

# Table ------------------------------------------------------------------------
tbl <- data.frame("Type" = "AllSubsets",  
                  "r squared" = .59, "RMSE Ratio" = rmseAlltest/rmseAlltr)
tbl <- tbl %>% rbind(list("Ridge", rsq$dev.ratio, rmseRidgeTest/rmseRidgeTrain)) 
tbl <- tbl %>% rbind(list("Lasso", rsqL$dev.ratio, rmseLassoTest/rmseLassoTrain)) 
tbl <- tbl %>% rbind(list("Relaxed Lasso", rsqLR$dev.ratio, rmseLassoTestR/rmseLassoTrainR)) 
tbl <- tbl %>% rbind(list("Lasso into Ridge", rsqL$dev.ratio, rmseRidgeTestL/rmseRidgeTrainL)) 
tbl <- tbl %>% rbind(list("Elastic Net", rsqEL$dev.ratio,rmseTestE/rmseTrainE)) 
tbl <- tbl %>% rbind(list("PCA components", .56, rmsePCtest/rmsePCtr))
tbl

plot(tbl$r.squared, col = tbl$Type)
barplot(tbl$r.squared, names.arg = tbl$Type, ylab = "R Squared")
barplot(tbl$RMSE.Ratio, names.arg = tbl$Type, ylab = "Testing Performance", ylim = c(0,1.2))
