import streamlit as st
import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

from .ClassifyNB import classify
from .class_vis import prettyPicture, output_image
from .prep_terrain_data import makeTerrainData

def write():
    st.title('Speed Determination based on Slope and Terrain')
    features_train, labels_train, features_test, labels_test = makeTerrainData()
    ### the training data (features_train, labels_train) have both "fast" and "slow" points mixed
    ### in together--separate them so we can give them different colors in the scatterplot,
    ### and visually identify them
    grade_fast = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii]==0]
    bumpy_fast = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii]==0]
    grade_slow = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii]==1]
    bumpy_slow = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii]==1]
    clf = classify(features_train, labels_train)
    prettyPicture(clf, features_test, labels_test)
    st.write("Creating a Decision surface based on a dummy dataset of feature grade and bumpiness")
    #------------------------------------------------------------------#
    st.subheader("Calculating NB Accuracy ")
    st.subheader("using the trained classifier to predict labels for the test features")
    st.write(" Method 1")
    accuracy = clf.score(features_test, labels_test)
    st.subheader(" Accuracy = {}".format(accuracy))
    st.write(" Method 2")  
    pred = clf.predict(features_test)
    accuracy = accuracy_score(pred, labels_test)
    st.subheader(" Accuracy = {}".format(accuracy))
