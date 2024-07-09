import os
import streamlit as st
import pandas as pd
from data_preprocessing import preprocess_data
from knowledge_representation import visualize_data
from patterns_insights import main as patterns_and_insights_main
from trained_models import display_trained_models
from train_models import train_and_save_models

def data_description(df):
    """this function displays the data """
    st.write("# Data Description")
    st.write("""
    ### Description:
    This dataset captures user engagement metrics from an online course platform, facilitating analyses on factors influencing course completion. It includes user demographics, course-specific data, and engagement metrics.

    ### Features:

    1. **UserID**: Unique identifier for each user
    2. **CourseCategory**: Category of the course taken by the user (e.g., Programming, Business, Arts)
    3. **TimeSpentOnCourse**: Total time spent by the user on the course in hours
    4. **NumberOfVideosWatched**: Total number of videos watched by the user
    5. **NumberOfQuizzesTaken**: Total number of quizzes taken by the user
    6. **QuizScores**: Average scores achieved by the user in quizzes (percentage)
    7. **CompletionRate**: Percentage of course content completed by the user
    8. **DeviceType**: Type of device used by the user (Device Type: Desktop (0) or Mobile (1))
    9. **CourseCompletion (Target Variable)**: Course completion status (0: Not Completed, 1: Completed)

    ### Target:

    **Distribution of the Target Variable (CourseCompletion)**:
    - 0 (Not Completed): 48%
    - 1 (Completed): 52%
    """)
    st.write("""### Top few rows""")
    st.write(df.head())

def main():
    st.title("InsightNexus: Knowledge Representation and Insight Generation")
    st.sidebar.title("MENU")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_file_path = os.path.join(script_dir, '..', 'data', 'online_course_engagement_data.csv')

    try:
        df = pd.read_csv(default_file_path)
        df1 = df
    except FileNotFoundError:
        st.error("Dataset file not found. Please ensure 'online_course_engagement_data.csv' is placed in the 'data' directory.")
        return

    df, label_encoders = preprocess_data(df)

    options = st.sidebar.radio("Choose an option", [
        "Data Description",
        "Data Visualization",
        "Train Models",
        "Trained Models",
        "Patterns and Insights"
    ])

    if options == "Data Description":
        data_description(df1)
    elif options == "Data Visualization":
        visualize_data(df)
    elif options == "Train Models":
        train_and_save_models(df)
        st.success("Models trained and saved successfully. Please go to 'Trained Models' to view the results.")
    elif options == "Trained Models":
        display_trained_models(df)
    elif options == "Patterns and Insights":
        patterns_and_insights_main(df)

if __name__ == "__main__":
    main()
