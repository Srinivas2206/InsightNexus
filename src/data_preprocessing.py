import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

def preprocess_data(df):
    # Drop UserID column as we know it is not a significant column from EDA
    df = df.drop(columns=["UserID"])
    # Encode categorical features
    label_encoders = {}
    categorical_features = ["CourseCategory", "DeviceType", "CourseCompletion"]
    for col in categorical_features:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

    return df, label_encoders

def split_and_scale_data(df):
    # Let's define X and Y
    X = df.drop(columns=["CourseCompletion"])
    y = df["CourseCompletion"]
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # Scaling the data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test
