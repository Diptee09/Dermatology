import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import numpy as np

# Set page configuration
st.set_page_config(page_title="Dermatology Disease Classifier", layout="wide")

# Load Dataset
@st.cache_data
def load_data():
    df = pd.read_csv('C:/Users/DELL/Desktop/Dermatology_App/dermatology.data', header=None)

    # Replace '?' with NaN
    df = df.replace('?', np.nan)

    # Convert to numeric
    df = df.apply(pd.to_numeric)

    # Fill missing values with column mean
    df = df.fillna(df.mean())

    return df

df = load_data()

# Prepare features and target
X = df.iloc[:, :-1]  # Features
y = df.iloc[:, -1]   # Target

# Load the trained model
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

# Sidebar Navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Dataset", "Summary", "Graph", "Predict"])

# Home Page
if page == "Home":
    st.title("🩺 Dermatology Disease Classifier")
    st.write("Welcome to the Dermatology Disease Classification App!")
    st.write("Use the sidebar to explore dataset, graphs, and make predictions.")

# Dataset Page
elif page == "Dataset":
    st.title("📄 Dataset")
    st.write("Here’s a preview of the dataset:")
    st.write(df.head())

# Summary Page
elif page == "Summary":
    st.title("📊 Dataset Summary")
    st.write(df.describe())

# Graph Page
elif page == "Graph":
    st.title("📈 Graphs")

    st.subheader("Target Class Distribution")
    fig, ax = plt.subplots()
    sns.countplot(x=y, ax=ax)
    ax.set_xlabel("Disease Class")
    ax.set_ylabel("Count")
    ax.set_title("Disease Class Distribution")
    st.pyplot(fig)

# Predict Page
elif page == "Predict":
    st.title("🔮 Disease Prediction")

    st.write("Enter feature values to predict the disease class:")

    user_input = []
    for i in range(X.shape[1]):
        value = st.number_input(f'Feature {i + 1}', min_value=0.0, value=0.0)
        user_input.append(value)

    if st.button('Predict'):
        input_df = pd.DataFrame([user_input], columns=X.columns)
        prediction = model.predict(input_df)[0]
        st.success(f'Predicted Disease Class: {prediction}')
