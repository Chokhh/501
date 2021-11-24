import pandas as pd
from sklearn.naive_bayes import MultinomialNB
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix,roc_auc_score,roc_curve
import matplotlib.pyplot as plt
from sklearn.svm import SVC, LinearSVC
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.inspection import permutation_importance
import seaborn as sns


df=pd.read_csv("/Users/balogZ/Desktop/501/501 Assignment 5/text/DT_text.csv", index_col=False)
df=df.drop(df.columns[0], axis=1)

x = df.drop(columns=['sector'])
y = df.sector 

## split into training and testing data
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1)


#################
## Naive Bayes ##
#################
nb = MultinomialNB()
nb.fit(X_train, y_train)
predicted = nb.predict(X_test)
print(classification_report(y_test,predicted ))


####  Confusion matrix ####
label = ['Energy','Financial','Retailing','Technology','Health Care']
conf_matrix = confusion_matrix(y_test,predicted,labels=label )

sns.heatmap(conf_matrix.T,square=True, annot=True, fmt='d', cbar=True, cmap="YlGnBu",
                    xticklabels=set(label),
                    yticklabels=set(label))
plt.xlabel('Target')
plt.ylabel('predicted')



####  Feature importance  ####
imps1=permutation_importance(nb, X_test, y_test)
NB_importances = imps1.importances_mean
std1 = imps1.importances_std
indices = np.argsort(NB_importances)[::-1]
feature_names=X_train.columns
feat_importances = pd.Series(NB_importances, index=feature_names)
feat_importances.nlargest(17).plot(kind='barh')



















