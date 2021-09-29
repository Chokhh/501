import pandas as pd

# Add age to the dataframe from the corpus

fortune1000=pd.read_csv("/Users/balogZ/Desktop/501 Assignment 2/R cleaned.csv")
ceo_names=fortune1000["CEO"]

df=pd.read_csv("/Users/balogZ/Desktop/501 Assignment 2/Python cleaned.csv")
df["birth_year"]=""


for i in range(1000):
    name=str(ceo_names[i])
    wikipage=open("/Users/balogZ/Desktop/wikipages/"+name+".txt")
    string=wikipage.read()
    birth=string.partition(")")[0][-4:]
    df["birth_year"][i]=birth


df["birth_year"][2]=1965


# Clean the birth_year column

for i in range(1000):
    if str(df["birth_year"][i])[:-2] != '19':
        df["birth_year"][i]=''

# Calculate age from birth_year

df["age"]=""

for i in range(1000):
    if str(df["birth_year"][i]) != '':
        df["age"][i]= 2021-int(df["birth_year"][i])-1
    


# Write to new csv

df.to_csv("/Users/balogZ/Desktop/501 Assignment 2/age added.csv", index=False)



