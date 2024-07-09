import streamlit as st
from joblib import load
from data_preprocessing import split_and_scale_data
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import plotly.express as px


def display_trained_models(df):
    st.write("## Trained Models")

    X_train, X_test, y_train, y_test = split_and_scale_data(df)

    models = ["logistic_regression_model.joblib", "svc_model.joblib", "decision_tree_model.joblib",
              "adaboost_model.joblib", "gradient_boosting_model.joblib", "random_forest_model.joblib",
              "xgboost_model.joblib", "kneighbors_model.joblib"]

    for model_file in models:
        try:
            model = load(f'../models/{model_file}')
            y_pred = model.predict(X_test)
            st.write(f"### {model_file.split('.')[0].capitalize()}")
            st.write(f"Accuracy: {accuracy_score(y_test, y_pred)}")
            st.write("Classification Report:")
            st.text(classification_report(y_test, y_pred))
            cm = confusion_matrix(y_test, y_pred)
            fig = px.imshow(cm, text_auto=True, title=f"{model_file.split('_')[0].capitalize()} Confusion Matrix")
            st.plotly_chart(fig)
        except FileNotFoundError:
            st.error(f"Model file {model_file} not found. Please ensure the model is trained and saved correctly.")
