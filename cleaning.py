# ==============================
# DATA CLEANING & PREPROCESSING
# ==============================

# Step 1: Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Step 2: Load Dataset
df = pd.read_csv("Titanic-Dataset.csv")

print("First 5 Rows:")
print(df.head())

print("\nDataset Info:")
print(df.info())

print("\nMissing Values:")
print(df.isnull().sum())


# Step 3: Handle Missing Values

# Fill Age with Median
df['Age'].fillna(df['Age'].median(), inplace=True)

# Fill Embarked with Mode
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)

# Drop Cabin column (too many missing values)
df.drop(columns='Cabin', inplace=True)

print("\nAfter Handling Missing Values:")
print(df.isnull().sum())


# Step 4: Convert Categorical → Numerical

# Convert Sex to numbers
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})

# One-hot encoding for Embarked
df = pd.get_dummies(df, columns=['Embarked'], drop_first=True)

print("\nAfter Encoding:")
print(df.head())


# Step 5: Feature Scaling (Standardization)
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

num_cols = ['Age', 'Fare', 'SibSp', 'Parch']
df[num_cols] = scaler.fit_transform(df[num_cols])


# Step 6: Detect Outliers using Boxplot
plt.figure(figsize=(8,5))
sns.boxplot(data=df[['Age', 'Fare']])
plt.title("Boxplot Before Removing Outliers")
plt.show()


# Step 7: Remove Outliers using IQR Method
def remove_outliers(data, column):
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    
    return data[(data[column] >= lower) & (data[column] <= upper)]

df = remove_outliers(df, 'Age')
df = remove_outliers(df, 'Fare')


# Step 8: Final Output
print("\nFinal Dataset Shape:", df.shape)
print(df.describe())


# Step 9: Save Cleaned Dataset
df.to_csv("Cleaned_Titanic.csv", index=False)

print("\n✅ Data Cleaning Completed — Cleaned file saved as Cleaned_Titanic.csv")