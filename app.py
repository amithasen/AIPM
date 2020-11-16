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
	<div style="background-color:tomato;padding:3px">
	<h2 style="color:white;text-align:center;"> AIPM Dashboard </h2>
	</div>
	"""
st.markdown(html_temp,unsafe_allow_html=True)

st.markdown("### Risers Team Members : Sridhar, Ram, Iqbal, Rex & Senthilnathan ")

##st.header('AIPM Dashboard')
st.write(' Welcome to our AIPM Dashboard! here you can see how '
         ' we can make use of AI & ML feature to accomplish Management goals/vision.')


#########################################################################################
st.markdown("### Speech Recognition")

import speech_recognition as sr
import requests
import os.path
import base64

#import urllib2

is_check = st.checkbox("Generate Audio to Text file")
if is_check:
    r=sr.Recognizer()
    path = st.file_uploader("Select a audio file", type=['wav', 'mp3'])
    if path is not None:

        demo=sr.AudioFile(path)

        with demo as source:
                audio=r.record(source)

        a=r.recognize_google(audio, language='en-IN')

        def download_link(object_to_download, download_filename, download_link_text):
            if isinstance(object_to_download,pd.DataFrame):
                object_to_download = object_to_download.to_csv(index=False)

            # some strings <-> bytes conversions necessary here
            b64 = base64.b64encode(object_to_download.encode()).decode()

            return f'<a href="data:file/txt;base64,{b64}" download="{download_filename}">{download_link_text}</a>'


        #st.write(a)

        if st.button('Generate text file'):
            tmp_download_link = download_link(a, 'Speech_Recognized_File.txt', 'Click here to download your text file!')
            st.markdown(tmp_download_link, unsafe_allow_html=True)


        #path= os.path.join(os.environ['USERPROFILE'], 'Desktop')

        #path= os.path.expanduser(os.sep.join(['~','Desktop']))

        #filename= "Speech_Recognized_File"

        #full_path= os.path.join(path,filename+".txt")

        #file = open(full_path,"w") 
        #file.write(a) 
        #file.close() 

            st.markdown('Your Speech Recognition file is generated successfully!')
#else:
#    st.markdown("## No Audio File Selected!!!")


#########################################################################################

st.markdown("### Functional Graphs & Visualization")


page = st.sidebar.selectbox(
    "Functional Graphs & Visualization",
    ("Effort Distribution Summary", "Sprint Execution Summary", "Resource Utilization Summary")
)

if page == "Effort Distribution Summary":

    #st.markdown("### Effort Distribution Summary")

    df = pd.read_excel("Rally_Sprint.xlsx")

    df["Effort"] = df["Formatted ID"].str.slice(0,2)

    Effort = { "Effort" : {"US": "Scope", "DE": "Defects"}}
    df.replace(Effort, inplace=True)

    plt.figure(figsize=(6,6))

    Graph1 = pd.value_counts(df["Effort"]).plot(kind="pie", autopct='%1.0f%%', title = 'Effort Distribution Summary', legend =True)
    st.write(Graph1)
    st.pyplot()

    #import matplotlib.pyplot as plt
    #%matplotlib inline
    #pd.value_counts(df["Effort"]).plot(kind="pie", autopct='%1.1f%%')
        
elif page == "Sprint Execution Summary":

    st.markdown("### Sprint Execution Summary")

    df = pd.read_excel("Rally_Sprint.xlsx")

    df_trends = df[["Accepted Date","Creation Date","Iteration","Schedule State"]]

    options = ["Released",'Accepted']
    df_trends = df_trends[df_trends["Schedule State"].isin(options)]

    import datetime as dt

    df_trends['Accepted Date'] = pd.to_datetime(df_trends['Accepted Date']).dt.date
    df_trends['Creation Date'] = pd.to_datetime(df_trends['Creation Date']).dt.date

    df_trends["Days"] = df_trends["Accepted Date"] - df_trends["Creation Date"]

    df_trends["Days"]= df_trends["Days"].dt.days.astype('int16')

    a = df_trends.groupby(['Iteration'],sort=False).agg({'Days': ['min', 'max']})
    b = df_trends.groupby(['Iteration'],sort=False).agg({'Days': ['mean', 'std']})

    result = df_trends.groupby(['Iteration'],sort=False).agg({'Days': ['min', 'max', 'mean', 'std']})

    from pandas.plotting import table
    #ax= result.plot(kind='barh', title='Sprint Execution Summary', figsize=[12,6], legend=True, stacked=True)
    ax= a.plot(kind='bar', title='Sprint Execution Trends', figsize=[12,6],legend=True, stacked=True, colormap='winter')
    b.plot(ax=ax, kind='line', legend=True)

    ax.set_ylabel('Days', fontsize=9)

    Graph2 = table(ax, np.round(result,0))
    st.write(Graph2)
    st.pyplot()

else:
    st.markdown("## This Graph is Under Construction!!!")

##############################################################################################################

#uploaded_file = st.file_uploader("Choose your data file", type="csv")
#if uploaded_file is not None:
	#df = pd.read_csv(uploaded_file)
	#st.write(data)
page = st.sidebar.selectbox(
    "Type of Project Management Template/tool to use?",
    ("Jira", "Rally", "Microsoft PM", "Others")
)

#page = st.sidebar.selectbox("Choose a page", ['Homepage', 'Exploration', 'Prediction'])
if page == 'Jira':

    st.markdown("### Exploratory Data Analysis")

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
    st.markdown("### Interactive Chart")
    ## select multi-columns
    columns = st.multiselect(label='Choose the columns you want to display from below drop down!', options=df.columns)
    st.area_chart(df[columns])  

    ## to display correlation heatmap

    # get correation of each feature in dataset
    corrmat=df.corr()
    top_corr_features=corrmat.index
    plt.figure(figsize=(16,16))
    # plot the heatmap
    hmap=sns.heatmap(df[top_corr_features].corr(),annot=True,cmap="RdYlGn")

    is_check = st.checkbox("Display Correlation Heatmap")
    if is_check:
        st.write(hmap)
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

    st.markdown("### Model Summary")

    is_check = st.checkbox("Our Model Accuracy Summary")
    if is_check:
        st.write('Model Accuracy Score:',rfcl.score(X_test , y_test))
        st.write('Model Precision Score:',metrics.precision_score(y_test,Y_predict))
        st.write('Model Recall Score:',metrics.recall_score(y_test,Y_predict))
        st.write('Model Confusion Matrix:')
        st.write(metrics.confusion_matrix(y_test, Y_predict))

elif page == 'Rally':

    st.markdown("### Exploratory Data Analysis")

    df = pd.read_excel("Rally_Sprint.xlsx")

    ## to display raw data
    is_check = st.checkbox("Display Raw Data")
    if is_check:
        st.write(df)

    df["Milestones"]= df["Milestones_Month"] + df["Milestone_Week"]

    ## to display classification in train dataset
    is_check = st.checkbox("Classification in the training set")
    if is_check:
        chart_data = pd.value_counts(df["Milestones"])
        st.bar_chart(chart_data)
    
    ## To trasform target column to UTC format
    cleanup_milestones = { "Milestones" : {"Feb Week 2" : "08/02/2020", "Mar Week 2" : "14/03/2020", "Apr Week 1" : "04/04/2020", "May Week 2" : "09/05/2020", "May Week 4" : "23/05/2020", "Jun Week 1" : "06/06/2020", "Jun Week 2" : "13/06/2020", "Sep Week 1" : "05/09/2020"}}
    df.replace(cleanup_milestones, inplace=True)


    df["Milestones"]= pd.to_datetime(df["Milestones"]) 

    df["Milestones"] = pd.to_datetime(df['Milestones'].values).astype(int)/ 10**9

    df['Milestones'] = df['Milestones'].astype(int)


    ## Dropping unwanted or irrelavant columns..

    df = df.drop('State', axis=1)
    df = df.drop('Acceptance Criteria', axis=1)
    df = df.drop('Biz Priority', axis=1)
    df = df.drop('Business Application Name', axis=1)
    df = df.drop('Closed Date', axis=1)
    df = df.drop('Defect Status', axis=1)
    df = df.drop('Defect Type', axis=1)
    df = df.drop('Defects', axis=1)
    df = df.drop('Defect Suites', axis=1)
    df = df.drop('Duplicates', axis=1)
    df = df.drop('ERMO Release Name', axis=1)
    df = df.drop('Feature', axis=1)
    df = df.drop('HasParent', axis=1)
    df = df.drop('Parent', axis=1)
    df = df.drop('Portfolio Item', axis=1)
    df = df.drop('Priority', axis=1)
    df = df.drop('Opened Date', axis=1)
    df = df.drop('Resolution', axis=1)
    df = df.drop('RootCause', axis=1)
    df = df.drop('Tags', axis=1)
    df = df.drop('Test Case', axis=1)
    df = df.drop('Target Date', axis=1)
    df = df.drop('Target Build', axis=1)
    df = df.drop('Milestones_Month', axis=1)
    df = df.drop('Milestone_Week', axis=1)
    df = df.drop('Blocked', axis=1)
    df = df.drop('Project', axis=1)
    df = df.drop('Ready', axis=1)

    ## converting all categorical variable to numerical columns

    for feature in df.columns: # Loop through all columns in the dataframe
        if df[feature].dtype == 'float64': # Only apply for columns with categorical strings
           df[feature] = df[feature].astype(str) # Replace strings with an integer

    ## converting all categorical variable to numerical columns

    for feature in df.columns: # Loop through all columns in the dataframe
        if df[feature].dtype == 'object': # Only apply for columns with categorical strings
            df[feature] = pd.Categorical(df[feature]).codes # Replace strings with an integer

    ## converting all categorical variable to numerical columns

    for feature in df.columns: # Loop through all columns in the dataframe
        if df[feature].dtype == 'bool': # Only apply for columns with categorical strings
            df[feature] = pd.Categorical(df[feature]).codes # Replace strings with an integer
        
    st.markdown("### Interactive Chart")
    ## select multi-columns
    columns = st.multiselect(label='Choose the columns you want to display from below drop down!', options=df.columns)
    st.line_chart(df[columns])

    # get correation of each feature in dataset
    corrmat=df.corr()
    top_corr_features=corrmat.index
    plt.figure(figsize=(16,16))
    # plot the heatmap
    hmap=sns.heatmap(df[top_corr_features].corr(),annot=True,cmap="RdYlGn")

    is_check = st.checkbox("Display Correlation Heatmap")
    if is_check:
        st.write(hmap)
        st.pyplot()

    #from scipy.stats import zscore
    #df_z = df.apply(zscore)

    X = df.drop("Milestones" , axis=1)
    y= df["Milestones"]

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.30, random_state=1)

    from sklearn.ensemble import  GradientBoostingRegressor
    from sklearn.ensemble import  RandomForestRegressor

    from sklearn import metrics
    from sklearn.metrics import accuracy_score
    from sklearn.metrics import confusion_matrix,classification_report

    rfTree = RandomForestRegressor(n_estimators=100)
    rfTree = rfTree.fit(X_train,y_train)

    Y_predict = rfTree.predict(X_test)


    ## to display accuracy of our model

    st.markdown("### Model Summary & Prediction")

    is_check = st.checkbox("Our Model Accuracy Summary")
    if is_check:
        st.write('Model Accuracy Score in train data:', rfTree.score(X_train, y_train))
        st.write('Model Accuracy Score in test:', rfTree.score(X_test, y_test))

    is_check = st.checkbox("Regression model output")
    if is_check:
        st.write('Regression Model output in test data:', pd.to_datetime(Y_predict, unit='s'))

else:
    st.markdown("# Page Under Construction!")

########################################################################################

add_selectbox = st.sidebar.selectbox(
    "Type of Machine learning you want to use?",
    ("Supervised", "Unsupervised", "Reinforcement")
)

add_selectbox = st.sidebar.selectbox(
    "Which AI/ML model you want to use?",
    ("Regression", "Classification", "Clustering", "Neural Networks")
)

add_selectbox = st.sidebar.selectbox(
    "Which Algorithm you want to use?",
    ("Linear Regression","Logistic Regression","Naive Bayes","SVM","KNN","Decision Tree","RandomForest","ANN","CNN","RNN")
)

# add_selectbox = st.sidebar.selectbox(
#     "Functional Graphs & Visualization",
#     (" ", "Graph 1", "Graph 2", "Graph 3")
# )

    
###########################################################################################

#link = '[GitHub](http://github.com)'
#st.markdown(link, unsafe_allow_html=True)
#This will open the link when clicked on it.

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


#with st.file_input() as input:
#  if input == None:
#   st.warning('No file selected.')
#  else:
#    file_contents = input.read()

###########################################################################################

#st.markdown("## Party time!")
#st.write("Yay! We're done with our analysis and prediction. Click below to celebrate.")
#btn = st.button("Let's Celebrate!")
#if btn:
#    st.balloons()
