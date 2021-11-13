library(plyr)
library(ggplot2)
library(rpart)
library(rattle)
library("rpart.plot")
library(caret)
library(forcats)

DF <- read.csv("/Users/balogZ/Desktop/501/501 Assignment 3/record.csv")
df<-DF[DF$sector %in% c('Financials', 'Energy', 'Technology', 'Retailing', 'Health Care'),]
df$sector<-as.factor(df$sector)

set.seed(2021)

index <- sample(2,nrow(df),replace = T,prob=c(0.7,0.3))#70% for training model
trainset<-df[index==1,]
testset<-df[index==2,]

#remove the label from the test data set
test_label <- testset$sector
testset <-  testset[ , -which(names(testset) %in% c("sector"))]


####################################    DT1    ################################################

# plot
DT <- rpart(trainset$sector ~ ., data = trainset, method="class")
DT
rpart.plot(DT)


# confusion matrix & accuracy
DT_Prediction= predict(DT, testset, type="class")

confusionMatrix(table(DT_Prediction, test_label))
out <- table(DT_Prediction, test_label) ## one way to make a confu mat
out <- data.frame(out)
out
cofm1 <- ggplot(data = out, aes(x = test_label, y = DT_Prediction))
cofm1 <- cofm1 + geom_tile(aes(fill = Freq))
cofm1 <- cofm1 + scale_fill_gradient(low = '#56B1F7', high = '#132B43')
cofm1 <- cofm1 + geom_text(aes(label = Freq))
cofm1

confMat<-table(test_label,DT_Prediction)
accuracy<-sum(diag(confMat))/sum(confMat)
accuracy


# importance
importance <- data.frame(DT$variable.importance)
colnames(importance) <- c('importance')
importance$feature <- rownames(importance)

importance %>%
  mutate(feature = fct_reorder(feature, desc(importance))) %>%
  ggplot(aes(x = feature, y = importance, fill = feature)) + 
  geom_bar(stat = 'identity') +
  theme(axis.text.x=element_text(angle=55,hjust=1, vjust=1))


####################################    DT2    ################################################

# plot
DT2 <- rpart(trainset$sector ~ ., data = trainset, method="class",
             cp=0.01,parms = list(split="information"),minsplit=2)
DT2
rpart.plot(DT2)


# confusion matrix & accuracy
DT_Prediction2= predict(DT2, testset, type="class")

confusionMatrix(table(DT_Prediction2, test_label))
out2 <- table(DT_Prediction2, test_label) ## one way to make a confu mat
out2 <- data.frame(out2)
out2
cofm2 <- ggplot(data = out2, aes(x = test_label, y = DT_Prediction2))
cofm2 <- cofm2 + geom_tile(aes(fill = Freq))
cofm2 <- cofm2 + scale_fill_gradient(low = '#56B1F7', high = '#132B43')
cofm2 <- cofm2 + geom_text(aes(label = Freq))
cofm2

confMat2<-table(test_label,DT_Prediction2)
accuracy2<-sum(diag(confMat2))/sum(confMat2)
accuracy2


# importance
importance2 <- data.frame(DT2$variable.importance)
colnames(importance2) <- c('importance')
importance2$feature <- rownames(importance2)

importance2 %>%
  mutate(feature = fct_reorder(feature, desc(importance))) %>%
  ggplot(aes(x = feature, y = importance, fill = feature)) + 
  geom_bar(stat = 'identity') +
  theme(axis.text.x=element_text(angle=55,hjust=1, vjust=1))


####################################    DT3    ################################################

# plot
DT3 <- rpart(trainset$sector ~ num..of.employees+profit.margin+rank+rank_change, 
             data = trainset, method="class", cp=0)
DT3
rpart.plot(DT3)


# confusion matrix & accuracy
DT_Prediction3= predict(DT3, testset, type="class")

confusionMatrix(table(DT_Prediction3, test_label))
out3 <- table(DT_Prediction3, test_label) ## one way to make a confu mat
out3 <- data.frame(out3)
out3
cofm3 <- ggplot(data = out3, aes(x = test_label, y = DT_Prediction3))
cofm3 <- cofm3 + geom_tile(aes(fill = Freq))
cofm3 <- cofm3 + scale_fill_gradient(low = '#56B1F7', high = '#132B43')
cofm3 <- cofm3 + geom_text(aes(label = Freq))
cofm3

confMat3<-table(test_label,DT_Prediction3)
accuracy3<-sum(diag(confMat3))/sum(confMat3)
accuracy3


# importance
importance3 <- data.frame(DT3$variable.importance)
colnames(importance3) <- c('importance')
importance3$feature <- rownames(importance3)

importance3 %>%
  mutate(feature = fct_reorder(feature, desc(importance))) %>%
  ggplot(aes(x = feature, y = importance, fill = feature)) + 
  geom_bar(stat = 'identity') +
  theme(axis.text.x=element_text(angle=55,hjust=1, vjust=1))
