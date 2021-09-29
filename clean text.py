import wikipedia
import pandas as pd
import csv

fortune1000=pd.read_csv("/Users/balogZ/Desktop/501 Assignment 2/R cleaned.csv")
ceo_names=fortune1000["CEO"]

def wikipage():
    for i in range(1000):
        name=str(ceo_names[i])
        wikipage=open("/Users/balogZ/Desktop/wikipages/"+name+".txt", "w")
        try:
            content=wikipedia.page(title=name,
                     pageid=None, auto_suggest=True,
                     redirect=True, preload=False).content
            wikipage.write(content)
            wikipage.close()
        except:
            wikipage.write("NA")
            wikipage.close()

# wikipage()


def replace():
    for i in range(1000):
        name=str(ceo_names[i])
        wikipage=open("/Users/balogZ/Desktop/wikipages/"+name+".txt")
        words=wikipage.read()
        if words=="NA":
            wikipage=open("/Users/balogZ/Desktop/wikipages/"+name+".txt", "w")
            try:
                content=wikipedia.page(title=name,
                         pageid=None, auto_suggest=False,
                         redirect=False, preload=False).content
                wikipage.write(content)
                wikipage.close()
            except:
                wikipage.write("NA")
                wikipage.close()
        
# replace()




new=pd.read_csv("/Users/balogZ/Desktop/501 Assignment 2/Python cleaned.csv")
ceo_names_no_middle = new["name"]




def wikipage_no_middle():
    for i in range(1000):
        name=str(ceo_names[i])
        name_no_middle=str(ceo_names_no_middle[i])
        wikipage=open("/Users/balogZ/Desktop/wikipages/"+name+".txt")
        words=wikipage.read()
        if words=="NA":
            wikipage=open("/Users/balogZ/Desktop/wikipages/"+name+".txt", "w")
            try:
                content=wikipedia.page(title=name_no_middle,
                         pageid=None, auto_suggest=False,
                         redirect=False, preload=False).content
                wikipage.write(content)
                wikipage.close()
            except:
                wikipage.write("NA")
                wikipage.close()

# wikipage_no_middle()


def replace_no_middle():
    for i in range(1000):
        name=str(ceo_names[i])
        name_no_middle=str(ceo_names_no_middle[i])
        wikipage=open("/Users/balogZ/Desktop/wikipages/"+name+".txt")
        words=wikipage.read()
        if words=="NA":
            wikipage=open("/Users/balogZ/Desktop/wikipages/"+name+".txt", "w")
            try:
                content=wikipedia.page(title=name_no_middle,
                         pageid=None, auto_suggest=True,
                         redirect=True, preload=False).content
                wikipage.write(content)
                wikipage.close()
            except:
                wikipage.write("NA")
                wikipage.close()

replace_no_middle()
                
