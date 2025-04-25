import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")  # optional for clean plots

# Load the dataset
df = pd.read_csv('Titanic-Dataset.csv')

# Display first 5 rows
print("First 5 Rows:")
print(df.head())

# Shape of the dataset
print("\nShape of the dataset:")
print(df.shape)

# Column names
print("\nColumn Names:")
print(df.columns)

# Info about dataset
print("\nDataset Info:")
print(df.info())

# Summary statistics for numeric columns
print("\nSummary Statistics:")
print(df.describe())

# Histograms for all numeric columns
df.hist(figsize=(12, 10), edgecolor='black')
plt.tight_layout()
plt.show()
plt.figure(figsize=(12, 6))
sns.boxplot(data=df.select_dtypes(include='number'))
plt.xticks(rotation=45)
plt.title("Boxplot of Numeric Features")
plt.show()
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', linewidths=0.5)
plt.title("Correlation Heatmap")
plt.show()

# Histograms for all numeric columns
df.hist(figsize=(12, 10), edgecolor='black')
plt.tight_layout()
plt.show()

# Boxplot for numeric features
plt.figure(figsize=(12, 6))
sns.boxplot(data=df.select_dtypes(include='number'))
plt.xticks(rotation=45)
plt.title("Boxplot of Numeric Features")
plt.show()

# Correlation heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', linewidths=0.5)
plt.title("Correlation Heatmap")
plt.show()








