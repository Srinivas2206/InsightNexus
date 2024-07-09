import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import streamlit as st


def visualize_data(df):
    st.write("# Data Visualization")
    st.write("## Count Plots")
    # List of columns for count plots
    count_plot_columns = ["CourseCompletion", "DeviceType", "CourseCategory"]
    for column in count_plot_columns:
        st.write(f"### {column}")
        # Count plot using seaborn
        plt.figure(figsize=(10, 6))
        sns.countplot(data=df, x=column, palette="Set2")
        plt.title(f"Count Plot of {column}")
        plt.xlabel(column)
        plt.ylabel("Count")
        st.pyplot(plt.gcf())
        plt.clf()
        # Pie chart using plotly
        fig = px.pie(df, names=column, title=f"Pie Chart of {column}",
                     color_discrete_sequence=px.colors.qualitative.Set3)
        st.plotly_chart(fig)
    ''' 
    st.write("## Scatter Plots")

    # List of pairs of columns for scatter plots
    scatter_plot_pairs = [
        ("TimeSpentOnCourse", "CompletionRate"),
        ("NumberOfVideosWatched", "QuizScores"),
        ("NumberOfQuizzesTaken", "QuizScores")
    ]

    for x_col, y_col in scatter_plot_pairs:
        st.write(f"### {x_col} vs {y_col}")
        fig = px.scatter(df, x=x_col, y=y_col, color="CourseCompletion", title=f"Scatter Plot of {x_col} vs {y_col}")
        st.plotly_chart(fig)'''

    st.write("## Box Plots")

    # List of columns for box plots
    box_plot_columns = ["TimeSpentOnCourse", "QuizScores", "CompletionRate"]

    for column in box_plot_columns:
        st.write(f"### {column}")
        plt.figure(figsize=(10, 6))
        sns.boxplot(data=df, y=column, palette="Set2")
        plt.title(f"Box Plot of {column}")
        plt.ylabel(column)
        st.pyplot(plt.gcf())
        plt.clf()
