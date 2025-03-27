import streamlit as st
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.
import numpy as np

st.write("""
   # Explore different classifier 
   ## Which one is the best?
""")

dataset_name = st.sidebar.selectbox("Select Dataset", ("Iris", "Breast Cancer", "Wine dataset"))

classifier_name = st.sidebar.selectbox("Select Classifier", ("KNN", "SVM", "Random Forest"))

def get_dataset(dataset_name):
    if dataset_name == "Iris":
        data = datasets.load_iris()
    elif dataset_name == "Breast Cancer":
        data = datasets.load_breast_cancer()
    else:
        data = datasets.load_wine()
    x, y = data.data, data.target
    return x, y

def add_parameter_ui(classifier_name):
    parameter_dictionary = {}
    if classifier_name == "KNN":
        classifier = st.sidebar.slider("K", 1, 15)
        parameter_dictionary["K"] = classifier
    elif classifier_name == "SVM":
        classifier = st.sidebar.slider("C", 0.01, 10.0)
        parameter_dictionary["C"] = classifier
    else:
        max_depth = st.sidebar.slider("max_depth", 2, 15)
        no_of_estimators = st.sidebar.slider("no_of_estimators", 1, 100)
        parameter_dictionary["max_depth"] = max_depth
        parameter_dictionary["no_of_estimators"] = no_of_estimators
    return parameter_dictionary

def get_classifier(classifier_name, parameter_dictionary):


x, y = get_dataset(dataset_name)

st.write(f"Shape of dataset: {x.shape}")
st.write(f"Number of classes: {np.unique_values(y)}")

parameter_dictionary = add_parameter_ui(classifier_name)
