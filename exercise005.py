# Re-import necessary libraries and reload the dataset
# pip install scikit-learn 
# pip install pandas
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier
import streamlit as st

# Reload the Titanic dataset
titanic_data = pd.read_csv("processed_titanic_data.csv")

# Split the data into features (X) and target (y)
X = titanic_data.drop("Survived", axis=1)
y = titanic_data["Survived"]

# Train a Random Forest classifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X, y)

# Get feature importances
feature_importances = rf_classifier.feature_importances_

# Combine feature names and their importance scores
features = list(X.columns)
feature_importance_dict = dict(zip(features, feature_importances))

# Sort features based on importance
sorted_features = sorted(feature_importance_dict.items(), key=lambda x: x[1], reverse=True)

# Get top 3 features
top_3_features = sorted_features[:4]
st.write(top_3_features)

