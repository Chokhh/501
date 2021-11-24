library(caTools)
library(naivebayes)
library(dplyr)
library(ggplot2)
library(e1071)
library(cvms)
library(caret)

DF <- read.csv("/Users/balogZ/Desktop/501/501 Assignment 3/record.csv")
df<-DF[DF$sector %in% c('Financials', 'Energy', 'Technology', 'Retailing', 'Health Care'),]
df$sector<-as.factor(df$sector)


## Normalize the data
min_max_norm <- function(x) {
  (x - min(x)) / (max(x) - min(x))
}


norm_df=as.data.frame(lapply(df[, -1],min_max_norm))
norm_df=cbind(df$sector,norm_df)
names(norm_df)[1]<-'sector'

#write.csv(norm_df, "norm_record.csv")
norm_df <- read.csv("/Users/balogZ/Desktop/record/norm_record.csv")
norm_df<-norm_df[, -1]
norm_df$sector<-as.factor(norm_df$sector)
## split into train/test data
set.seed(2021)

index <- sample(2,nrow(norm_df),replace = T,prob=c(0.75,0.25)) #75% for training model
trainset<-norm_df[index==1,]
testset<-norm_df[index==2,]

## remove the label from both datasets
test_label <- testset$sector
testset<-  testset[ , -which(names(testset) %in% c("sector"))]

train_label <- trainset$sector
trainset_N <-  trainset[ , -which(names(trainset) %in% c("sector"))]



# Naive Bayes
NB <- naive_bayes(trainset_N, train_label, laplace = 1) # using laplace smoothing
plot(NB)


# prediction
y_pred <- predict(NB, testset)
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

# feature importance
train <- trainset
model = train(sector ~ .,data=train,method="naive_bayes")
X= varImp(model)
plot(X)


