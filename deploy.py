# -*- coding: utf-8 -*-
"""
Created on Fri Jul  8 02:44:13 2022

@author: rakesh
"""

import pandas as pd
import numpy as np
import sklearn
import pickle
import streamlit as st
import os
import random
from sklearn.feature_extraction.text import CountVectorizer
vect = CountVectorizer(stop_words='english') 
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
nltk.download('punkt')
stop_words = set(stopwords.words("english"))
#loading the model
#loaded_model=pickle.load(open('‪‪C:/Users/rakesh/trained_model.sav', 'rb'))



import os
folder_path=r'C:\Users\rakesh\Downloads\exercise_task-1\exercise_task\data\train'
fileName=os.listdir(folder_path)
data=[]
for i in fileName:
  text=os.path.abspath(os.path.join(folder_path,i))
  fileName01=os.listdir(text)
  for j in fileName01:
    text01=os.path.abspath(os.path.join(text,j))
    abc=open(text01,"r")
    with open(text01,'r') as files:
      try:
        for k in files:
          fgh=[]
          fgh.append(k)
          fgh.append(i)
          data.append(fgh) 
      except:
        p=1    

df = pd.DataFrame(data, columns=['content','lable'])
print(df)


count_row = df.shape[0] 
for i in range(0,count_row):
  if df['content'][i]!=None:
    if len(df['content'][i])==1:
      if ord(df['content'][i])==10:
        df=df.drop(axis=0,index=i)

print(df)
x = df['content'].values
y = df['lable'].values


# Split the data       
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,random_state=0)

from sklearn.feature_extraction.text import CountVectorizer  


vect = CountVectorizer(stop_words='english') 
x_train_vect = vect.fit_transform(x_train)
x_test_vect = vect.transform(x_test)

from sklearn.svm import SVC  
model = SVC()
model.fit(x_train_vect,y_train)
y_pred = model.predict(x_test_vect)



from sklearn.metrics import accuracy_score,classification_report
print(accuracy_score(y_pred,y_test))

print(classification_report(y_pred,y_test))

def produce():
    folder_path02=r'C:/Users/rakesh/Downloads/exercise_task-1/exercise_task/data'
    kl=1
    lis=[]
    while kl<5:
        fileName02=os.listdir(folder_path02)
        item02 = random.choice(list(fileName02))
        folder_path=os.path.abspath(os.path.join(folder_path02,item02))
        fileName=os.listdir(folder_path)
        item = random.choice(list(fileName))
        text=os.path.abspath(os.path.join(folder_path,item))
        fileName01=os.listdir(text)
        item01 = random.choice(list(fileName01))
        text01=os.path.abspath(os.path.join(text,item01))
        abc=open(text01,"r")
        a=abc.read()

        data=[a]

        df = pd.DataFrame(data, columns=['Numbers'])
        var=df['Numbers'][0]
        test = vect.transform([var])
        var1=model.predict(test)[0]                                                #the randomly choosen text file from (train + test) dataset is predicted by the above ExtraTreesClassifier model 
        lis.append(var1)
        if kl==1:
            var003=var1
            print("1) THE TOPIC OF THE PARAGRAPH IS",var1)
            print(a) 
            kl=kl+1 
            temp1=a
        else:
            count1 = lis.count(lis[0])
            if count1==1:
                var004=var1
                print("-----------------------------------------------------------------------------------------------")
                print("2) THE TOPIC OF THE PARAGRAPH IS",var1)
                print(a)
                temp2=a
                break
            else:
                lis.pop(-1)
        if len(lis)>=2:
            break   
    return temp1,temp2,var003,var004    

#produce()
#print(produce())
def finding(text08):
    test = vect.transform([text08])
    var01=model.predict(test)[0] 
    return var01
    

def main():
    
    st.title('COMPUTER-BASED ENGLISH LANGUAGE TEST SOFTWARE')
    st.text('please click on ender button to get two different interesting topic passages')
    #xyz=st.text_input('enter the text')
    if st.button("enter"):
        
        #st.write(finding(xyz))
        a,b,c,d=produce()
        st.text("1) the first topic belongs to")
        st.write("----->",c)
        st.write(a)
        st.text("----------------------------------------------------------------------------")
        
        st.text("2) the second topic belongs to ")
        st.write("----->",d)
        st.write(b)
        
        
        
if __name__ == '__main__':
    main()   
