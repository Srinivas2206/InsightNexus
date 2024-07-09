import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from mlxtend.frequent_patterns import fpgrowth, association_rules


def descriptive_statistics(df):
    st.write("## Descriptive Statistics")

    st.write("### Central Tendency")
    st.write("Mean:")
    st.write(df.mean())
    st.write("Median:")
    st.write(df.median())
    st.write("Mode:")
    st.write(df.mode().iloc[0])

    st.write("### Dispersion")
    st.write("Range:")
    st.write(df.max() - df.min())
    st.write("Variance:")
    st.write(df.var())
    st.write("Standard Deviation:")
    st.write(df.std())

    st.write("### Distribution")
    st.write("Skewness:")
    st.write(df.skew())
    st.write("Kurtosis:")
    st.write(df.kurtosis())


def correlation_analysis(df):
    st.write("## Correlation Analysis")
    corr_matrix = df.corr()
    fig, ax = plt.subplots()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', ax=ax)
    st.pyplot(fig)


def comparative_analysis(df):
    st.write("## Comparative Analysis")

    categorical_columns = df.select_dtypes(include=['category', 'object']).columns
    numerical_columns = df.select_dtypes(include=['number']).columns

    for cat_col in categorical_columns:
        st.write(f"### Analysis of {cat_col}")

        for num_col in numerical_columns:
            fig, ax = plt.subplots()
            sns.boxplot(x=cat_col, y=num_col, data=df, ax=ax)
            st.pyplot(fig)

            st.write(f"ANOVA result for {cat_col} and {num_col}:")
            anova_result = stats.f_oneway(*(df[df[cat_col] == cat][num_col] for cat in df[cat_col].unique()))
            st.write(anova_result)


def regression_analysis(df):
    st.write("## Regression Analysis")

    target = 'CompletionRate'  # Change this to your target variable
    features = df.drop(columns=[target]).columns

    X = df[features]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    st.write("### Regression Model")
    st.write(f"Coefficients: {model.coef_}")
    st.write(f"Intercept: {model.intercept_}")
    st.write(f"Mean Squared Error: {mean_squared_error(y_test, y_pred)}")


def outlier_detection(df):
    st.write("## Outlier Detection")

    numerical_columns = df.select_dtypes(include=['number']).columns
    z_scores = np.abs(stats.zscore(df[numerical_columns]))

    outliers = (z_scores > 3).any(axis=1)
    st.write(f"Number of outliers detected: {outliers.sum()}")


def kmeans_clustering(df):
    st.write("## Segment Analysis: K-Means Clustering")

    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df.select_dtypes(include=['number']))

    kmeans = KMeans(n_clusters=3, random_state=42)
    clusters = kmeans.fit_predict(scaled_data)

    df['Cluster'] = clusters

    fig, ax = plt.subplots()
    sns.pairplot(df, hue='Cluster', palette='viridis', diag_kind='kde')
    st.pyplot(fig)


def association_rules_analysis(df):
    st.write("## Association Rules")

    # Assuming binary data for market basket analysis
    basket = df.select_dtypes(include=['category', 'object'])

    basket_encoded = pd.get_dummies(basket)
    frequent_itemsets = fpgrowth(basket_encoded, min_support=0.1, use_colnames=True)
    rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.0)

    st.write(frequent_itemsets)
    st.write(rules)


def main(df):
    st.write("# Patterns and Insights")

    descriptive_statistics(df)
    correlation_analysis(df)
    comparative_analysis(df)
    regression_analysis(df)
    outlier_detection(df)
    #kmeans_clustering(df)
    #association_rules_analysis(df)
