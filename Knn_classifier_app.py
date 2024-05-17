import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st 
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
   

def plot_cv_results(K_range, cv_scores):
    plt.figure(figsize=(10, 6))
    plt.scatter(K_range, cv_scores, marker='o')
    plt.xlabel('Number of Neighbors K')
    plt.ylabel('Cross-Validated Accuracy')
    plt.title('k-NN Varying number of neighbors')
    st.pyplot(plt)
    print(f"Shape of K_range: {K_range}")
    print(f"Shape of cv_scores: {cv_scores}")


st.title('KNN Classifier Optimization')
st.write('Dataset preview')
df = pd.read_csv('Social_Network_Ads.csv')
st.write(df.head())

features = st.multiselect('select features to display for', options=df.columns.tolist())
label = st.selectbox('select label:',options=df.columns.tolist())

if features and label:
    X = df[features].values
    y = df[label].values
    
    label_encoders = {}
    for i , col in enumerate(features):
       if df[col].dtype == 'object':
        le = LabelEncoder()
        X[:, i] = le.fit_transform(X[:, i])
        label_encoders[col] = le

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    k_min = st.number_input('Mininum K ', min_value = 1 , max_value = 10)
    k_max = st.number_input('Maximum K', min_value = 1 , max_value = 15)
    if k_min >= k_max:
        st.error('Maximum K must be greater than minimum K')
    else:
        k_range = range(k_min , k_max + 1 )
        # K_range_list = list(K_range)

    cv_scores = []
    for k in k_range:
          knn = KNeighborsClassifier(n_neighbors=k)
          scores = cross_val_score(knn ,X,y,cv = 10, scoring = 'accuracy')
          cv_scores.append(scores.mean())
    
    k_range_list = list(k_range)
           
    if len(k_range_list) == len(cv_scores):

            optimal_k = k_range[np.argmax(cv_scores)]
            st.write(f'the optimal number of neighbors is',{optimal_k})

            plot_cv_results(k_range_list, cv_scores)
    else:
            st.error("Error in cross-validation: k_range_list and cv_scores length mismatch.")
else:
    st.warning("Please select both features and target variable.")