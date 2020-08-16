import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures,StandardScaler
from sklearn.pipeline import Pipeline
def write():
    st.write('House Sales in King County, USA')
    st.write('This dataset contains house sale prices for King County, which includes Seattle. It includes homes sold between May 2014 and May 2015.')