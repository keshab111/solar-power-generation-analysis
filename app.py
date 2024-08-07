import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset
@st.cache_data
def load_data():
    data = pd.read_csv('solarpowergeneration.csv')
    return data

data = load_data()

st.title('Solar Power Generation Analysis')

# Data Inspection
st.subheader('Data Inspection')
if st.checkbox('Show raw data'):
    st.write(data)

st.subheader('Summary Statistics')
st.write(data.describe())

# Handle missing values
mean_wind_speed = data['average-wind-speed-(period)'].mean()
data['average-wind-speed-(period)'].fillna(mean_wind_speed, inplace=True)

# Exploratory Data Analysis
st.subheader('Exploratory Data Analysis')

# Plot histograms
st.write('Histograms')
columns = data.columns
fig, axs = plt.subplots(len(columns)//3 + 1, 3, figsize=(15, 10))
axs = axs.flatten()

for ax, col in zip(axs, columns):
    ax.hist(data[col], bins=30)
    ax.set_title(col)

plt.subplots_adjust(hspace=0.5, wspace=0.5)  # Adjust the spacing here
st.pyplot(fig)

st.write('Pair Plot')
fig = sns.pairplot(data[['distance-to-solar-noon', 'temperature', 'wind-direction', 'wind-speed', 'sky-cover', 'power-generated']])
fig.fig.subplots_adjust(top=0.95, bottom=0.05, left=0.05, right=0.95, hspace=0.5, wspace=0.5)  # Adjust the spacing here
st.pyplot(fig)

st.write('Box Plot')
fig, ax = plt.subplots(figsize=(10, 6))
sns.boxplot(data=data, ax=ax)
plt.subplots_adjust(top=0.95, bottom=0.05, left=0.05, right=0.95, hspace=0.5, wspace=0.5)  # Adjust the spacing here
plt.xticks(rotation=90)
st.pyplot(fig)

st.write('Correlation Matrix')
corr_matrix = data.corr()
fig, ax = plt.subplots(figsize=(10, 6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', ax=ax)
plt.subplots_adjust(top=0.95, bottom=0.05, left=0.05, right=0.95, hspace=0.5, wspace=0.5)  # Adjust the spacing here
st.pyplot(fig)

# Model Building
st.subheader('Model Building and Evaluation')

# Splitting the data
X = data.drop(columns=['power-generated'])
y = data['power-generated']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model selection
model_options = ['Linear Regression', 'Random Forest', 'Gradient Boosting', 'SVR', 'KNN', 'Decision Tree', 'XGBoost']
selected_model = st.selectbox('Select a model', model_options)

# Model training and evaluation
if st.button('Train and Evaluate'):
    if selected_model == 'Linear Regression':
        model = LinearRegression()
    elif selected_model == 'Random Forest':
        model = RandomForestRegressor()
    elif selected_model == 'Gradient Boosting':
        model = GradientBoostingRegressor()
    elif selected_model == 'SVR':
        model = SVR()
    elif selected_model == 'KNN':
        model = KNeighborsRegressor()
    elif selected_model == 'Decision Tree':
        model = DecisionTreeRegressor()
    elif selected_model == 'XGBoost':
        model = XGBRegressor()

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    st.write(f'Model: {selected_model}')
    st.write(f'Mean Squared Error: {mse}')
    st.write(f'R-squared: {r2}')

    # Scatter plot of actual vs predicted values
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(y_test, y_pred, alpha=0.5)
    ax.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red')
    ax.set_xlabel('Actual values')
    ax.set_ylabel('Predicted values')
    ax.set_title(f'Actual vs Predicted values: {selected_model}')
    plt.subplots_adjust(top=0.95, bottom=0.05, left=0.05, right=0.95, hspace=0.5, wspace=0.5)  # Adjust the spacing here
    st.pyplot(fig)
