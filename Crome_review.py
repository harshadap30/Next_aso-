#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import streamlit as st
# Title for streamlit
st.title("Chrome Reviews - positive reviews with low ratings")


import pandas as pd


df = pd.read_csv('C:\Users\Harsha\Downloads\crome_reviews.csv')
#df.head()


# Here we are going to drop off unwanted columns from the dataframe, and also remove those rows that are totally empty
import numpy as np
df_NA = df.dropna(how = 'all')
#df[df['Star'] != 3]
#df_NA.keys()
df_NA['Positivity'] = np.where(df_NA['Star'] > 3, 1, 0)
cols = ['ID', 'Star', 'Review URL', 'Thumbs Up', 'User Name', 'Developer Reply', 'Version','Review Date', 'App ID']
df_NA.drop(cols, axis=1, inplace=True)


# we will do some NLP to clean the text, convert text to a corpus of words which will be used to evaluate the review
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer


# Initialize empty array
# to append clean text
corpus = []
num = len(df_NA)

df_NA['Text'] = df_NA['Text'].astype(str)
# (reviews) rows to clean
for i in range(0, num):
	
	# column : "Text", row ith
	review = re.sub('[^a-zA-Z]', ' ', df_NA['Text'][i])
	
	# convert all cases to lower cases
	review = review.lower()
	
	# split to array(default delimiter is " ")
	review = review.split()
	
	# creating PorterStemmer object to
	# take main stem of each word
	ps = PorterStemmer()
	
	# loop for stemming each word
	# in string array at ith row
	review = [ps.stem(word) for word in review
				if not word in set(stopwords.words('english'))]
				
	# rejoin all string array elements
	# to create back into a string
	review = ' '.join(review)
	
	# append each string to create
	# array of clean text
	corpus.append(review)

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1000)
X = cv.fit_transform(corpus).toarray()
y = df_NA.iloc[:, 1].values

# train a random forest model to perform classification as positive or negative review
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators = 350,criterion = 'entropy')
                             
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# confusion matrix lets us know the percentage of good predictions vs the wrong ones
from sklearn.metrics import confusion_matrix
 
cm = confusion_matrix(y_test, y_pred)


# using streamlit for accepting user input file as testcase
uploaded_file = st.file_uploader("Choose a file for checking review/rating discrepancy")
st.write("Waiting for input")
if uploaded_file is not None:
    
     
     df_test = pd.read_csv(uploaded_file)
 
# same what we did for training data - repeating it for test data
df_tNA = df_test.dropna(how = 'all')
df_tNA['Positivity'] = np.where(df_tNA['Star'] > 3, 1, 0)
cols = ['ID', 'Review URL', 'Thumbs Up', 'User Name', 'Developer Reply', 'Version','Review Date', 'App ID']
df_tNA.drop(cols, axis=1, inplace=True)



# NLP stuff on test file
corpus_test = []
num = len(df_tNA)

df_tNA['Text'] = df_tNA['Text'].astype(str)
# (reviews) rows to clean
for i in range(0, num):
	
	# column : "Text", row ith
	review = re.sub('[^a-zA-Z]', ' ', df_tNA['Text'][i])
	
	# convert all cases to lower cases
	review = review.lower()
	
	# split to array(default delimiter is " ")
	review = review.split()
	
	# creating PorterStemmer object to
	# take main stem of each word
	ps = PorterStemmer()
	
	# loop for stemming each word
	# in string array at ith row
	review = [ps.stem(word) for word in review
				if not word in set(stopwords.words('english'))]
				
	# rejoin all string array elements
	# to create back into a string
	review = ' '.join(review)
	
	# append each string to create
	# array of clean text
	corpus_test.append(review)
#print(corpus_test)

cv_test = CountVectorizer(max_features = 1000)
X_testr = cv_test.fit_transform(corpus_test).toarray()
y_testr = df_tNA.iloc[:, 2].values

# prediction using the same model trained before using random forest
y_predr = model.predict(X_testr)
cm = confusion_matrix(y_testr, y_predr)

# printing all the reviews which are supposedly good but having a bad rating
st.write("The list of reviews where the reviews and ratings probably don't match are as below")
for i in range(0, len(df_tNA)):
    if(y_predr[i] == 1 and y_testr[i] == 0):
            st.write(df_tNA['Text'][i],df_tNA['Star'][i])
            #count = count + 1

