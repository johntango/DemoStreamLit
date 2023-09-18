import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# read in train.csv
train = pd.read_csv('Titanic_train.csv')
st.write("Titanic Data")

st.write(train)

# Clean the data
# Get the top 5 features most correlated with 'Survived'


