import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix,accuracy_score
import streamlit as st 
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler


st.title('KNN Classifier')
# st.write('Dataset preview')
df = pd.read_excel('D:\Divya Dharshini\Praxis\AML\Titanic dataset (1).xlsx')
st.write(df.head())
   

df['Age'].fillna(df['Age'].median(), inplace=True)
# data['Embarked'].fillna(data['Embarked'].mode()[0], inplace=True)
df['Fare'].fillna(df['Fare'].median(), inplace=True)
# df.drop(columns=['Class', 'Name'], inplace=True)

le = LabelEncoder()
df['Gender'] = le.fit_transform(df['Gender'])



features = st.multiselect("Select features for the model:", options=df.columns.tolist(), default=['Class', 'Age', 'Fare', 'Gender' ])
# st.write("Selected features:")
# st.write(features)
if 'Survival' in features:
   features.remove('Survival')

X = df[features]
y = df['Survival']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
k = st.slider("Choose K value for KNN", 1, 20, 5)

model = KNeighborsClassifier(n_neighbors=k)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

# Define the confusion matrix
confusion_matrix = np.array([[134, 10],
                            [72, 46]])

# Extracting values
TN, FN,FP,TP = confusion_matrix.ravel()

# Printing the values
print(f"True Negatives (TN): {TN}")
print(f"False Positives (FP): {FP}")
print(f"False Negatives (FN): {FN}")
print(f"True Positives (TP): {TP}")


st.write(f"Accuracy: {accuracy:.2f}")
st.write("Confusion Matrix:")
st.write(cm)
