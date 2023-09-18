import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
'''
For linear regression analysis to predict survivability on the Titanic dataset, we need to process the data to make it suitable for modeling. Here's a step-by-step approach:

Handle Missing Data: Identify and impute or drop missing values.
Feature Engineering: Extract and create relevant features that could have an impact on the outcome.
Encode Categorical Variables: Convert categorical variables into a format that can be provided to machine learning algorithms (e.g., one-hot encoding).
Normalize Data: Scale features to ensure they have similar scales, which helps with the regression algorithm.
Remove Unnecessary Columns: Drop columns that aren't useful for the regression analysis.

'''
# read in train.csv
train = pd.read_csv('Titianic_train.csv')
st.write("Titanic Data")

st.write(train)

# Clean the data
# Get the top 5 features most correlated with 'Survived'


