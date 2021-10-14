import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans
from sklearn import datasets
from sklearn import preprocessing
import pylab as pl
from sklearn import decomposition
import seaborn as sns
import sklearn
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.cluster import AgglomerativeClustering
import scipy.cluster.hierarchy as hc
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
import plotly.express as px
import plotly.io as pio
pio.renderers.default='browser'





df = pd.read_csv('/Users/balogZ/Desktop/MyCleanCorpusData.csv')
df=df.drop('LABEL', axis=1)
np.random.seed(5)

########################
### k-means


## Elbow
sum_of_squared_distances = []
K = range(1, 100)

for k in K:
    model = KMeans(n_clusters=k, init='k-means++', max_iter=200, n_init=1)
    model = model.fit(df)
    sum_of_squared_distances.append(model.inertia_)

plt.plot(K, sum_of_squared_distances, 'bx-')
plt.xlabel('k')
plt.ylabel('Sum of squared distances')
plt.title('Elbow Method')
plt.show()

## Silhouette
Sih=[]
Cal=[]
k_range=range(2,6)

for k in k_range:
    k_means_n = KMeans(n_clusters=k)
    model = k_means_n.fit(df)
    Pred = k_means_n.predict(df)
    labels_n = k_means_n.labels_
    R1=metrics.silhouette_score(df, labels_n, metric = 'euclidean')
    R2=metrics.calinski_harabasz_score(df, labels_n)
    Sih.append(R1)
    Cal.append(R2)

print(Sih) 
print(Cal) 

fig1, (ax1, ax2) = plt.subplots(nrows=2, ncols=1)
ax1.plot(k_range,Sih)
ax1.set_title("Silhouette")
ax1.set_xlabel("")
ax2.plot(k_range,Cal)
ax2.set_title("Calinski_Harabasz_Score")
ax2.set_xlabel("k values")

########################







########################
### PCA

df_norm=(df-df.mean())/df.std
NumCols=df_norm.shape[1]

## Instantiated my own copy of PCA
My_pca = PCA(n_components=4)  ## I want the two prin columns

## Transpose it
new_df = My_pca.fit_transform(df_norm)    

Comps = pd.DataFrame(new_df,
                        columns=['PC%s' % _ for _ in range(4)]
                        )

########################
## Look at 2D PCA clusters
############################################

plt.figure(figsize=(12,12))
plt.scatter(Comps.iloc[:,0], Comps.iloc[:,1], s=100, color="green")


plt.xlabel("PC 1")
plt.ylabel("PC 2")
plt.title("Scatter Plot Clusters PC 1 and 2",fontsize=15)


plt.show()
         
###### PCA 3D
new_df=pd.DataFrame(new_df, columns=['comp1','comp2', 'comp3', 'comp4'])
fig=px.scatter_3d(new_df, x='comp1', y='comp2', z='comp3', color='comp4')
fig.show()
fig.write_html('/Users/balogZ/Desktop/pca 3d.html')




###############################################
##
##         DBSCAN
##
###############################################


MyDBSCAN = DBSCAN(eps=6, min_samples=2)

MyDBSCAN.fit_predict(df)
print(MyDBSCAN.labels_)





####### k=2
model = KMeans(n_clusters=2)
model.fit(new_df)
labels = model.labels_
new_df['label'] = labels 


db=DBSCAN(0.02)
db.fit(new_df)
labels=list(map(str,db.labels_))
fig=px.scatter_3d(new_df, x='comp1', y='comp2', z='comp3', color='label')
fig.show()
fig.write_html('/Users/balogZ/Desktop/k=2.html')

####### k=3
model = KMeans(n_clusters=3)
model.fit(new_df)
labels = model.labels_
new_df['label'] = labels 


db=DBSCAN(0.02)
db.fit(new_df)
labels=list(map(str,db.labels_))
fig=px.scatter_3d(new_df, x='comp1', y='comp2', z='comp3', color='label')
fig.show()
fig.write_html('/Users/balogZ/Desktop/k=3.html')

####### k=4
model = KMeans(n_clusters=4)
model.fit(new_df)
labels = model.labels_
new_df['label'] = labels 


db=DBSCAN(0.02)
db.fit(new_df)
labels=list(map(str,db.labels_))
fig=px.scatter_3d(new_df, x='comp1', y='comp2', z='comp3', color='label')
fig.show()
fig.write_html('/Users/balogZ/Desktop/k=4.html')




      
########################
### hierarchical clustering

plt.figure(figsize =(12, 12))
plt.title('Hierarchical Clustering')
dendro = hc.dendrogram((hc.linkage(df, method ='ward')))
plt.show()



# wordcloud
 for i in set(labels):
     clusterwords = []
     select_labels = Myresults[Myresults[1] == i][0].tolist()
     for index, columns in df.iterrows():
         if index in select_labels:
             for j in range(df.shape[1]):
                 word = df.columns.values[j]
                 for k in range(int(df.loc[index, word])):
                     clusterwords.append(str(df.columns.values[j]))
     clusterwords = ' '.join(clusterwords)
     print(clusterwords)
     wordcloud = WordCloud(collocations=False, background_color = 'black', scale = 1.5).generate(clusterwords)
     plt.imshow(wordcloud)
     plt.axis('off')
     plt.show()
     
     
