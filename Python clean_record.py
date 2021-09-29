import pandas as pd
import numpy as np
import re
import statistics as stat
import matplotlib.pyplot as plt

df=pd.read_csv("/Users/balogZ/Desktop/501 Assignment 2/R cleaned.csv")


# Drop the unnamed first column

df.drop('Unnamed: 0', inplace=True, axis=1) # axis=1 columns


# Create new column of ceo names without middle names

df['name']=df['CEO']

def remove_middle():
    list1=[]
    for i in range(1000):
        list1.append(re.sub(' [A.-Z.]* ', ' ', str(df['name'][i])))
                     
        df['name']=df['name'].replace(df['name'][i], list1[i])

remove_middle()
        

# Reorder the columns

df.set_index('CEO', inplace=True)

df=df[['name','company', 'rank', 'sector',
       'newcomer', 'ceo_founder', 'ceo_woman']]


df.to_csv("/Users/balogZ/Desktop/501 Assignment 2/Python cleaned.csv")
