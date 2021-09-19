import streamlit as st
import numpy as np
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()


def get_dataset(_data):
    if _data == 'Iris':
        data = datasets.load_iris()
    elif _data == 'Breast Cancer':
        data = datasets.load_breast_cancer()
    else:
        data = datasets.load_wine()
    X = data.data
    y = data.target
    return X, y


def add_param_ui(clf_name):
    params = dict()
    if clf_name == "KNN":
        K = st.sidebar.slider("K", 1, 15)
        params['K'] = K
    elif clf_name == "SVM":
        C = st.sidebar.slider("C", 0.01, 10.0)
        params['C'] = C
    else:
        max_depth = st.sidebar.slider("Max Depth", 2, 15)
        n_estimators = st.sidebar.slider("Number of Estimators", 1, 100)
        params['max_depth'] = max_depth
        params['n_estimators'] = n_estimators
    return params


def get_classifier(clf_name, params):
    if clf_name == "KNN":
        clf = KNeighborsClassifier(n_neighbors=params['K'])
    elif clf_name == "SVM":
        clf = SVC(C=params['C'])
    else:
        clf = RandomForestClassifier(
            n_estimators=params['n_estimators'], max_depth=params['max_depth'], random_state=9527)
    return clf


st.title("ML Dashboard")
st.write("""
# Explore different classifier
Which one is the best?
""")

dataset_name = st.sidebar.selectbox(
    "Select Dataset", ('Iris', 'Breast Cancer', 'Wine Dataset'))
classifier_name = st.sidebar.selectbox(
    "Select Classifier", ('KNN', 'SVM', 'Random Forest'))

##########################################################

X, y = get_dataset(dataset_name)
st.write("Dataset - ", dataset_name)
st.write("Shape", X.shape)
st.write("classes", len(np.unique(y)))


##########################################################
params = add_param_ui(classifier_name)
clf = get_classifier(classifier_name, params)

# classification process
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

st.write(f"Classifier - {classifier_name}")
st.write("Accuracy", accuracy)
st.write("---")
st.write("Scatter Plot")
# plot
pca = PCA(2)
X_projected = pca.fit_transform(X)
x1 = X_projected[:, 0]
x2 = X_projected[:, 1]
fig = plt.figure()
plt.scatter(x1, x2, c=y, alpha=0.8, cmap='viridis')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.colorbar()

st.pyplot(fig)
