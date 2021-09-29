from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem import PorterStemmer
import numpy as np
import pandas as pd
import os
import re


path="/Users/balogZ/Desktop/wikipages"

FileNameList=os.listdir(path)

MyFileNameList=[]
FileNames=[]

for nextfile in os.listdir(path):
    fullpath=path+"/"+nextfile
    MyFileNameList.append(fullpath)
    FileNames.append(nextfile)

MyCV=CountVectorizer(input='filename',
                        stop_words='english',
                        #max_features=100
                        )

My_DTM=MyCV.fit_transform(MyFileNameList)

MyColumnNames=MyCV.get_feature_names()

My_DF=pd.DataFrame(My_DTM.toarray(),columns=MyColumnNames)


CleanNames=[]
for filename in FileNames:
    ## remove the .txt from each file name
    newName=filename.rstrip(".txt")
    CleanNames.append(newName)

My_DF["LABEL"]=CleanNames

My_DF=My_DF.filter(regex='^\D')
My_DF=My_DF.filter(regex='\D$')

for i in My_DF.columns:
    column_sum=My_DF[i].sum()
    if column_sum <= 10:
        My_DF=My_DF.drop(columns=[i])
        

## Write to csv file

My_DF.to_csv('MyCleanCorpusData.csv', index=False)

