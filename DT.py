import pandas as pd
import random
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split 
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn import tree
import seaborn as sns
from sklearn.metrics import accuracy_score

import os
# os.environ["PATH"] += os.pathsep + '/opt/homebrew/lib/python3.9/site-packages/graphviz/'
os.environ["PATH"] += os.pathsep + '/opt/homebrew/bin'

# little bit cleaning on the dataset

df = pd.read_csv('/Users/balogZ/Desktop/501/501 Assignment 3/MyCleanCorpusData.csv')


df2=df

for i in df2.columns:
    column_sum=df2[i].sum()
    if column_sum <= 20:
        df2=df2.drop(columns=[i])
        

#df2.to_csv('/Users/balogZ/Desktop/text data.csv')



edu=pd.read_csv('/Users/balogZ/Desktop/501/501 Assignment 3/education.csv')
edu=edu[['CEO', 'sector']]
edu = edu.rename(columns = {"CEO":"LABEL"})

DF=pd.merge(df2, edu, on='LABEL')
DF = DF.rename(columns = {"sector_y":"sector"})

new_df=DF.loc[DF['sector'].isin(['Financials', 'Energy', 'Technology', 'Retailing', 'Health Care'] )]
new_df=new_df.drop('LABEL', axis=1)

#new_df.to_csv('/Users/balogZ/Desktop/DT_text.csv')



##### train/test
np.random.seed(2021)


topics=['Financials', 'Energy', 'Technology', 'Retailing', 'Health Care']

TrainDF, TestDF = train_test_split(new_df, test_size=0.3)

TestLabels=TestDF['sector']
TestDF = TestDF.drop(['sector'], axis=1)

TrainLabels=TrainDF["sector"]
TrainDF = TrainDF.drop(['sector'], axis=1)  


######################    Decision trees 1     ######################
MyDT = DecisionTreeClassifier(criterion='entropy',  ##"entropy" or "gini"
                              splitter='best',  ## or "random" or "best"
                              max_depth=6,
                              min_samples_split=2,
                              min_samples_leaf=1,
                              min_weight_fraction_leaf=0.0,
                              max_features=None,
                              random_state=None,
                              max_leaf_nodes=None,
                              min_impurity_decrease=0.0,
                              #min_impurity_split=None,
                              class_weight=None)

### fit the decision tree
MyDT.fit(TrainDF, TrainLabels)

feature_names = TrainDF.columns

Tree_Object = tree.export_graphviz(MyDT, out_file=None,
                                   ## The following creates TrainDF.columns for each
                                   ## which are the feature names.
                                   feature_names=feature_names,
                                   class_names=topics,
                                   filled=True, rounded=True,
                                   special_characters=True)

graph = graphviz.Source(Tree_Object)
graph.format='png'
graph.render("MyTree1")


## Confusion Matrix
print("Prediction\n")
DT_pred = MyDT.predict(TestDF)
print(DT_pred)

bn_matrix1 = confusion_matrix(TestLabels, DT_pred)
print("\nThe confusion matrix is:")
print(bn_matrix1)
fig = plt.figure()
sns_plot = sns.heatmap(bn_matrix1, annot = True, cmap='YlGnBu')
plt.ylabel('Predictions', fontsize=15)
plt.xlabel('Actuals', fontsize=15)
plt.savefig('conf_m1.png')
plt.show()

print('\nThe accuracy score is:')
accuracy_score(TestLabels, DT_pred)

## Importance
FeatureImp = MyDT.feature_importances_
indices = np.argsort(FeatureImp)[::-1]
## print out the important features.....
feature_sele_name = []
feature_sele_imp = []
for f in range(TrainDF.shape[1]):
    if FeatureImp[indices[f]] > 0:
        print("%d. feature %d (%f)" % (f + 1, indices[f], FeatureImp[indices[f]]))
        print("feature name: ", feature_names[indices[f]])
        feature_sele_imp.append(FeatureImp[indices[f]])
        feature_sele_name.append(feature_names[indices[f]])

### visualize the importance of different features
feat_importances = pd.Series(FeatureImp, index=feature_names)
feat_importances.nlargest(17).plot(kind='barh')
plt.ylabel('Features', fontsize=15)
plt.xlabel('Importance', fontsize=15)



######################    Decision trees 2     ######################
MyDT2 = DecisionTreeClassifier(criterion='entropy',  ##"entropy" or "gini"
                              splitter='best',  ## or "random" or "best"
                              max_depth=20,
                              min_samples_split=2,
                              min_samples_leaf=1,
                              min_weight_fraction_leaf=0.0,
                              max_features='sqrt',
                              random_state=None,
                              max_leaf_nodes=None,
                              min_impurity_decrease=0.0,
                              #min_impurity_split=None,
                              class_weight=None)

### fit the decision tree
MyDT2.fit(TrainDF, TrainLabels)

feature_names = TrainDF.columns

Tree_Object2 = tree.export_graphviz(MyDT2, out_file=None,
                                   ## The following creates TrainDF.columns for each
                                   ## which are the feature names.
                                   feature_names=feature_names,
                                   class_names=topics,
                                   filled=True, rounded=True,
                                   special_characters=True)

graph2 = graphviz.Source(Tree_Object2)
graph2.format='png'
graph2.render("MyTree2")


## Confusion Matrix
print("Prediction\n")
DT_pred2 = MyDT2.predict(TestDF)
print(DT_pred2)

bn_matrix2 = confusion_matrix(TestLabels, DT_pred2)
print("\nThe confusion matrix is:")
print(bn_matrix2)
fig = plt.figure()
sns_plot = sns.heatmap(bn_matrix2, annot = True, cmap='YlGnBu')
plt.ylabel('Predictions', fontsize=15)
plt.xlabel('Actuals', fontsize=15)
plt.savefig('conf_m2.png')
plt.show()

print('\nThe accuracy score is:')
accuracy_score(TestLabels, DT_pred2)

## Importance
FeatureImp2 = MyDT2.feature_importances_
indices = np.argsort(FeatureImp2)[::-1]
## print out the important features.....
feature_sele_name = []
feature_sele_imp = []
for f in range(TrainDF.shape[1]):
    if FeatureImp2[indices[f]] > 0:
        print("%d. feature %d (%f)" % (f + 1, indices[f], FeatureImp2[indices[f]]))
        print("feature name: ", feature_names[indices[f]])
        feature_sele_imp.append(FeatureImp2[indices[f]])
        feature_sele_name.append(feature_names[indices[f]])

### visualize the importance of different features
feat_importances2 = pd.Series(FeatureImp2, index=feature_names)
feat_importances2.nlargest(17).plot(kind='barh')
plt.ylabel('Features', fontsize=15)
plt.xlabel('Importance', fontsize=15)




######################    Decision trees 3     ######################
MyDT3 = DecisionTreeClassifier(criterion='gini',  ##"entropy" or "gini"
                              splitter='best',  ## or "random" or "best"
                              max_depth=10,
                              min_samples_split=2,
                              min_samples_leaf=1,
                              min_weight_fraction_leaf=0.0,
                              max_features=None,
                              random_state=None,
                              max_leaf_nodes=None,
                              min_impurity_decrease=0.0,
                              #min_impurity_split=None,
                              class_weight=None)

### fit the decision tree
MyDT3.fit(TrainDF, TrainLabels)

feature_names = TrainDF.columns

Tree_Object3 = tree.export_graphviz(MyDT3, out_file=None,
                                   ## The following creates TrainDF.columns for each
                                   ## which are the feature names.
                                   feature_names=feature_names,
                                   class_names=topics,
                                   filled=True, rounded=True,
                                   special_characters=True)

graph3 = graphviz.Source(Tree_Object3)
graph3.format='png'
graph3.render("MyTree3")


## Confusion Matrix
print("Prediction\n")
DT_pred3 = MyDT3.predict(TestDF)
print(DT_pred3)

bn_matrix3 = confusion_matrix(TestLabels, DT_pred3)
print("\nThe confusion matrix is:")
print(bn_matrix3)
fig = plt.figure()
sns_plot = sns.heatmap(bn_matrix3, annot = True, cmap='YlGnBu')
plt.ylabel('Predictions', fontsize=15)
plt.xlabel('Actuals', fontsize=15)
plt.savefig('conf_m3.png')
plt.show()

print('\nThe accuracy score is:')
accuracy_score(TestLabels, DT_pred3)

## Importance
FeatureImp3 = MyDT3.feature_importances_
indices = np.argsort(FeatureImp3)[::-1]
## print out the important features.....
feature_sele_name = []
feature_sele_imp = []
for f in range(TrainDF.shape[1]):
    if FeatureImp3[indices[f]] > 0:
        print("%d. feature %d (%f)" % (f + 1, indices[f], FeatureImp3[indices[f]]))
        print("feature name: ", feature_names[indices[f]])
        feature_sele_imp.append(FeatureImp3[indices[f]])
        feature_sele_name.append(feature_names[indices[f]])

### visualize the importance of different features
feat_importances3 = pd.Series(FeatureImp3, index=feature_names)
feat_importances3.nlargest(17).plot(kind='barh')
plt.ylabel('Features', fontsize=15)
plt.xlabel('Importance', fontsize=15)