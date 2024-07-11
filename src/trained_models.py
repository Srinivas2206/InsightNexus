import streamlit as st
from joblib import load
from data_preprocessing import split_and_scale_data
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import plotly.express as px
import pandas as pd

def display_trained_models(df):
    """ This function displays the performance of trained machine learning models on a Streamlit app.
    It loads each model, makes predictions on the test data, and displays the accuracy, classification report,
    and confusion matrix for each model.
    parameter:
    df(pandas.DataFrame)
    """
    st.write("## Trained Models")

    X_train, X_test, y_train, y_test = split_and_scale_data(df)

    models = ["logistic_regression_model.joblib", "svc_model.joblib", "decision_tree_model.joblib",
              "adaboost_model.joblib", "gradient_boosting_model.joblib", "random_forest_model.joblib",
              "xgboost_model.joblib", "kneighbors_model.joblib"]
    accuracies = []
    for model_file in models:
        try:
            model = load(f'../models/{model_file}')
            y_pred = model.predict(X_test)
            st.write(f"### {model_file.split('.')[0].capitalize()}")
            st.write(f"Accuracy: {accuracy_score(y_test, y_pred)}")
            accuracies.append(accuracy_score(y_test, y_pred))
            st.write("Classification Report:")
            st.text(classification_report(y_test, y_pred))
            cm = confusion_matrix(y_test, y_pred)
            fig = px.imshow(cm, text_auto=True, title=f"{model_file.split('_')[0].capitalize()} Confusion Matrix")
            st.plotly_chart(fig)
        except FileNotFoundError:
            st.error(f"Model file {model_file} not found. Please ensure the model is trained and saved correctly.")
    # Bar graph for accuracy scores
    st.subheader("Model Accuracy Comparison")
    accuracy_df = pd.DataFrame({
        "Model": models,
        "Accuracy": accuracies
    })
    fig = px.bar(accuracy_df, x="Model", y="Accuracy", title="Accuracy Scores of Trained Models",
                 labels={"Accuracy": "Accuracy Score", "Model": "Model"})
    st.plotly_chart(fig)
