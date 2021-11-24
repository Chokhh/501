library(caTools)
library(varImp)
library(klaR)
library(e1071)
library(caret)

df <- read.csv("/Users/balogZ/Desktop/norm_record.csv")
df<-df[, -1]
df$sector<-as.factor(df$sector)


## split into train/test data
set.seed(2021)

index <- sample(2,nrow(df),replace = T,prob=c(0.75,0.25)) #75% for training model
trainset<-df[index==1,]
testset<-df[index==2,]

## remove the label from testset
test_label <- testset$sector
testset <-  testset[ , -which(names(testset) %in% c("sector"))]





##################
## SVM kernel 1 ##  Linear
##################
tune.out=tune(svm ,sector~.,data=trainset ,kernel ="linear", 
              ranges =list(cost=c(0.001,0.01,0.1, 1,5,10,20,50,100,1000,10000)))
summary(tune.out)
bestmodel = tune.out$best.model
print(bestmodel)

# prediction
y_pred <- predict(bestmodel, testset)
(cm <- caret::confusionMatrix(y_pred, test_label))

# confusion metric
cmDF <- as.data.frame(cm$table)
plot_confusion_matrix(cmDF, 
                      target_col = "Reference", 
                      prediction_col = "Prediction",
                      counts_col = "Freq",
                      add_row_percentages = FALSE,
                      add_col_percentages = FALSE,
                      rm_zero_percentages = FALSE,
                      rm_zero_text = FALSE,
                      add_zero_shading = TRUE,
                      counts_on_top = TRUE)
# visualization
model <- svm(sector~.,data=trainset ,kernel ="linear", cost=10)
plt <- plot(model,trainset,plotType='scatter', num..of.employees ~ profit.margin)

# feature importance
train <- trainset
model = train(sector ~ .,data=trainset,method="svmLinear")
X= varImp(model)
plot(X)


##################
## SVM kernel 2 ##  Polynomial 
##################
tune.out=tune(svm ,sector~.,data=trainset ,kernel ="polynomial", 
              ranges =list(cost=c(0.001,0.01,0.1, 1,5,10,20,50,100,1000,10000)))
summary(tune.out)
bestmodel = tune.out$best.model
print(bestmodel)

# prediction
y_pred <- predict(bestmodel, testset)
(cm <- caret::confusionMatrix(y_pred, test_label))

# confusion metric
cmDF <- as.data.frame(cm$table)
plot_confusion_matrix(cmDF, 
                      target_col = "Reference", 
                      prediction_col = "Prediction",
                      counts_col = "Freq",
                      add_row_percentages = FALSE,
                      add_col_percentages = FALSE,
                      rm_zero_percentages = FALSE,
                      rm_zero_text = FALSE,
                      add_zero_shading = TRUE,
                      counts_on_top = TRUE)
# visualization
model <- svm(sector~.,data=trainset ,kernel ="polynomial", cost=10000)
plt <- plot(model,trainset,plotType='scatter', num..of.employees ~ profit.margin)

# feature importance
train <- trainset
model = train(sector ~ .,data=trainset,method="svmPoly")
X= varImp(model)
plot(X)



##################
## SVM kernel 3 ##  radial
##################
tune.out=tune(svm ,sector~.,data=train ,kernel ="radial", 
              ranges =list(cost=c(0.001,0.01,0.1, 1,5,10,20,50,100,1000,10000)))
summary(tune.out)
bestmodel = tune.out$best.model
print(bestmodel)

# prediction
y_pred <- predict(bestmodel, testset)
(cm <- caret::confusionMatrix(y_pred, test_label))

# confusion metric
cmDF <- as.data.frame(cm$table)
plot_confusion_matrix(cmDF, 
                      target_col = "Reference", 
                      prediction_col = "Prediction",
                      counts_col = "Freq",
                      add_row_percentages = FALSE,
                      add_col_percentages = FALSE,
                      rm_zero_percentages = FALSE,
                      rm_zero_text = FALSE,
                      add_zero_shading = TRUE,
                      counts_on_top = TRUE)
# visualization
model <- svm(sector~.,data=trainset ,kernel ="radial", cost=100)
plt <- plot(model,trainset,plotType='scatter', num..of.employees ~ profit.margin)

# feature importance
train <- trainset
model = train(sector ~ .,data=trainset,method="svmRadial")
X= varImp(model)
plot(X)
