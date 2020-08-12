#Importing needed libraries...
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
#import pandas_profiling as pp

st.set_option('deprecation.showfileUploaderEncoding', False)

hide_streamlit_style = """
            <style>
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)
#########################################################################################

#st.title('Testing Streamlit for our ML model')
html_temp = """
	<div style="background-color:tomato;padding:5px">
	<h2 style="color:white;text-align:center;"> AIPM Dashboard </h2>
	</div>
	"""
st.markdown(html_temp,unsafe_allow_html=True)

##st.header('AIPM Dashboard')
st.write('Welcome to our dashboard, here you can see how '
         ' we can make use of streamlit feature in our project dashboard.')

st.markdown("### Team members : Sridhar, Ram, Iqbal, Rex & Senthilnathan ")

#########################################################################################

add_selectbox = st.sidebar.selectbox(
    "Type of Project management template to use?",
    ("Jira", "Rally", "Microsoft PM", "Others")
)

add_selectbox = st.sidebar.selectbox(
    "Type of Machine learning you want to use?",
    ("Supervised", "Unsupervised", "Reinforcement")
)

add_selectbox = st.sidebar.selectbox(
    "Which Supervised ML you want to use?",
    ("Regression", "Classification", "Clustering", "Neural Networks")
)

add_selectbox = st.sidebar.selectbox(
    "Which Algorithm you want to use?",
    ("Linear Regression","Logistic Regression","Naive Bayes","SVM","KNN","Decision Tree","RandomForest","ANN","CNN","RNN")
)
###########################################################################################

uploaded_file = st.file_uploader("Choose your data file", type="csv")
if uploaded_file is not None:
	df = pd.read_csv(uploaded_file)
	st.write(data)

df = pd.read_excel("JIRA_subset2_final.xlsx")

## to display raw data
is_check = st.checkbox("Display Raw Data")
if is_check:
    st.write(df)

## to display classification in train dataset
is_check = st.checkbox("Classification in training set")
if is_check:
	chart_data = pd.value_counts(df["Resolution"])
	st.bar_chart(chart_data)


#is_check = st.checkbox("EDA on raw data")
#if is_check:
    #st.write(pp.ProfileReport(df))


## converting date fields to numeric ordinal values
######################################################################################

import datetime as dt

df['Created'] = pd.to_datetime(df['Created'])
df['Created'] = df['Created'].map(dt.datetime.toordinal)

df['Updated'] = pd.to_datetime(df['Updated'])
df['Updated'] = df['Updated'].map(dt.datetime.toordinal)

df['Due Date'] = pd.to_datetime(df['Due Date'])
df['Due Date'] = df['Due Date'].map(dt.datetime.toordinal)

df['Target Date'] = pd.to_datetime(df['Target Date'])
df['Target Date'] = df['Target Date'].map(dt.datetime.toordinal)

## converting all categorical variable to numerical columns

for feature in df.columns: # Loop through all columns in the dataframe
    if df[feature].dtype == 'object': # Only apply for columns with categorical strings
        df[feature] = pd.Categorical(df[feature]).codes # Replace strings with an integer

## dropping unwanted columns
df = df.drop('Due Date', axis=1)
df = df.drop('Resolution', axis=1)
df = df.drop('Issue Type', axis=1)
############################################################################################

## selected column to display
#column = st.selectbox('What column to you want to display', df.columns)
#st.line_chart(df[column])

## select multi-columns
columns = st.multiselect(label='What column to you want to display', options=df.columns)
st.line_chart(df[columns])

## to display correlation heatmap

is_check = st.checkbox("Display Correlation Heatmap")
if is_check:
    st.write(sns.heatmap(df.corr()))
    st.pyplot()
#############################################################################################

## Oversampling using SMOTE
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split

X = df.drop("Classification", axis=1)
y = df["Classification"] 
X_res, y_res = SMOTE().fit_sample(X, y)
test_size = 0.30 # taking 70:30 training and test set
seed = 7  # Random numbmer seeding for reapeatability of the code
X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=test_size, random_state=seed)


## Building the model
# Ensemble method - RandomForest Classifier..
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix,classification_report

rfcl = RandomForestClassifier(n_estimators = 30)
rfcl = rfcl.fit(X_train, y_train)

Y_predict = rfcl.predict(X_test)
##############################################################################################

## to display accuracy of our model

is_check = st.checkbox("Our Model Accuracy Summary")
if is_check:
    st.write('Model Accuracy Score:',rfcl.score(X_test , y_test))
    st.write('Model Precision Score:',metrics.precision_score(y_test,Y_predict))
    st.write('Model Recall Score:',metrics.recall_score(y_test,Y_predict))
    st.write('Model Confusion Matrix:')
    st.write(metrics.confusion_matrix(y_test, Y_predict))

#@st.cache
#def fetch_and_clean_data():
    #df = pd.read_csv('<some csv>')
    # do some cleaning
    #return df

#if st.button('Touch Me Not!'):
    #st.write('You shall not pass!')

#x = st.sidebar.slider('Select a value')
#st.write(x, 'squared is', x * x)

#if st.button('Predict'):
#model = pickle.load(open('model.pk', 'rb'))
#ypred = model.predict(df)
#st.write(ypred)

st.markdown("## Party time!")
st.write("Yay! We're done our ML web app using Streamlit. Click below to celebrate.")
btn = st.button("Celebrate!")
if btn:
    st.balloons()





