library(ggpubr)
library(factoextra)
library(cluster)
library(ggplot2)

#setwd('/Users/balogZ/Desktop/501 Assignment 3')
df<-read.csv('record.csv')
set.seed(2021)
data<-df[, c(2:10)]

# methods for optimal k
fviz_nbclust(data, kmeans, method = "wss")
fviz_nbclust(data, kmeans, method = "silhouette")
gap_cluster <- clusGap(data, kmeans, nstart = 25, K.max = 10, B = 1000)
fviz_gap_stat(gap_cluster)

 
# kmeans clustering

fit<-kmeans(data, 2)
fviz_cluster(fit, data = data, main = 'k=2')  
fit<-kmeans(data, 3)
fviz_cluster(fit, data = data, main = 'k=3')  
fit<-kmeans(data, 4)
fviz_cluster(fit, data = data, main = 'k=4')

s<- silhouette(fit$cluster, dist(data))
fviz_silhouette(s)

df$cluster = fit$cluster
write.csv(df, 'R clustered.csv', row.names=FALSE)

# Hierarchical clustering
hclust <- hcut(data, k = 3)

# visualization
fviz_cluster(fit, data = data)
fviz_dend(hclust)


d_eu <- dist(as.matrix(data), method = 'euclidean')
hclust <- hclust(d_eu)
plot(hclust, hang = -1, cex = 0.7)
rect.hclust(hclust, k = 3,)
d_man <- dist(data, method = 'manhattan')
mat_data <- as.matrix(data)
sim <- mat_data / sqrt(rowSums(mat_data * mat_data))
sim <- sim %*% t(sim)
