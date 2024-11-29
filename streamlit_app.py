import streamlit as st
import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

# File paths for saving the model and scaler
MODEL_FILE = "ann_model.pkl"
SCALER_FILE = "scaler.pkl"

# Load dataset
@st.cache_data
def load_data():
    data = load_breast_cancer()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df['target'] = data.target
    return df, data.feature_names, data.target_names

# Feature selection
@st.cache_data
def select_features(X, y, k=10):
    selector = SelectKBest(score_func=f_classif, k=k)
    X_new = selector.fit_transform(X, y)
    selected_features = np.array(feature_names)[selector.get_support()]
    return X_new, selected_features

# Train ANN model
def train_ann(X_train, y_train, params):
    model = MLPClassifier(
        hidden_layer_sizes=params['hidden_layer_sizes'],
        activation=params['activation'],
        solver=params['solver'],
        max_iter=params['max_iter']
    )
    model.fit(X_train, y_train)
    return model

# Save model and scaler to files
def save_model_and_scaler(model, scaler):
    with open(MODEL_FILE, "wb") as model_file:
        pickle.dump(model, model_file)
    with open(SCALER_FILE, "wb") as scaler_file:
        pickle.dump(scaler, scaler_file)
    st.success("Model and scaler saved successfully!")

# Load model and scaler from files
def load_model_and_scaler():
    try:
        with open(MODEL_FILE, "rb") as model_file:
            model = pickle.load(model_file)
        with open(SCALER_FILE, "rb") as scaler_file:
            scaler = pickle.load(scaler_file)
        return model, scaler
    except FileNotFoundError:
        return None, None

# Streamlit App
st.title("Breast Cancer Prediction App")

# Tabs for navigation
tab1, tab2, tab3, tab4 = st.tabs(["🏠 Home", "📊 Data Analysis", "🧠 Model Training", "🔮 Prediction"])

# Home Tab
with tab1:
    st.header("Welcome to the Breast Cancer Prediction App!")
    st.markdown(
        """
        This app allows you to:
        - Analyze the breast cancer dataset.
        - Train an Artificial Neural Network (ANN) model.
        - Make predictions based on user input.
        """
    )
    st.image("https://via.placeholder.com/800x400?text=Breast+Cancer+Prediction", caption="Breast Cancer Prediction")

# Data Analysis Tab
with tab2:
    st.header("Dataset and Analysis")
    df, feature_names, target_names = load_data()
    st.write("Dataset Overview:")
    st.dataframe(df.head())

    # Class distribution
    st.write("Class Distribution:")
    st.bar_chart(df['target'].value_counts())

    # Correlation heatmap
    st.write("Feature Correlation Heatmap:")
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(df.corr(), cmap="coolwarm", ax=ax, annot=False)
    st.pyplot(fig)

# Model Training Tab
with tab3:
    st.header("Train an ANN Model")

    # Feature Selection
    X = df.drop(columns=['target'])
    y = df['target']
    k = st.slider("Select number of features for training:", min_value=5, max_value=X.shape[1], value=10)
    X_selected, selected_features = select_features(X, y, k)
    st.write(f"Selected Features: {', '.join(selected_features)}")

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=42)

    # Scaling
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Model Hyperparameters
    st.subheader("Model Hyperparameters")
    hidden_layer_sizes = st.text_input("Hidden Layer Sizes (comma-separated):", "50,50")
    hidden_layer_sizes = tuple(map(int, hidden_layer_sizes.split(",")))
    activation = st.selectbox("Activation Function:", ["relu", "tanh", "logistic"])
    solver = st.selectbox("Solver:", ["adam", "sgd"])
    max_iter = st.number_input("Max Iterations:", min_value=100, max_value=1000, step=50, value=500)

    params = {
        "hidden_layer_sizes": hidden_layer_sizes,
        "activation": activation,
        "solver": solver,
        "max_iter": max_iter
    }

    # Train model
    if st.button("Train Model"):
        model = train_ann(X_train, y_train, params)
        save_model_and_scaler(model, scaler)

        # Evaluate the model
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        st.write(f"Model Accuracy: **{acc:.2f}**")

        # Classification Report
        st.subheader("Classification Report")
        st.text(classification_report(y_test, y_pred, target_names=target_names))

        # Confusion Matrix
        st.subheader("Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", xticklabels=target_names, yticklabels=target_names, ax=ax)
        plt.xlabel("Predicted")
        plt.ylabel("True")
        st.pyplot(fig)

# Prediction Tab
with tab4:
    st.header("Make a Prediction")

    # Load the model and scaler
    model, scaler = load_model_and_scaler()
    if model and scaler:
        user_input = {}
        for feature in selected_features:
            user_input[feature] = st.number_input(f"Enter value for {feature}:", value=0.0)
        user_data = np.array(list(user_input.values())).reshape(1, -1)
        user_data_scaled = scaler.transform(user_data)

        if st.button("Predict"):
            prediction = model.predict(user_data_scaled)
            prediction_proba = model.predict_proba(user_data_scaled)
            st.write(f"Prediction: **{target_names[prediction[0]]}**")
            st.write(f"Prediction Probability: {prediction_proba[0]}")
    else:
        st.warning("Model and scaler not found. Please train the model first.")
