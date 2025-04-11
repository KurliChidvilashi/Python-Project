#-----Performing EDA on Rainfall in India dataset-----#
#Loading dataset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_csv("D:\\Datasets\\Rainfall in india.csv")
print(df.info(),end="\n")
print(df.head(),end="\n")
print("Shape of dataset: ",df.shape)
print(df.describe())
print()

#Check for Missing Values
print(df.isnull().sum(),end = "\n")
missing_percent = (df.isnull().sum()/len(df))*100
print(missing_percent)
df.fillna(0, inplace=True)
print()

#Check for duplicates
print("Duplicates: ",df.duplicated().sum())
df = df.drop_duplicates()
print()

#Data Types and Unique Values
print("Subdivision: ",df["SUBDIVISION"].nunique())
print("Years:", df["YEAR"].nunique())
print()

#Outlier Detection using IQR
Q1 = df["ANNUAL"].quantile(0.25)
Q3 = df["ANNUAL"].quantile(0.75)
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

outliers = df[(df["ANNUAL"] < lower_bound) | (df["ANNUAL"] > upper_bound)]
print("Number of outliers in Annual Rainfall:", len(outliers))
print("Outliers removed:", len(df) - len(outliers))#To remove outliers
print()

#Skewness and Distribution
print("Skewness of Annual Rainfall:", df["ANNUAL"].skew())
sns.histplot(df["ANNUAL"], kde=True, color="purple")
plt.title("Annual Rainfall Distribution")
plt.show()
print()

#---Objective 1: Trend of Annual Rainfall Over Years---#
# Group by Year and calculate average rainfall
yearly_avg = df.groupby("YEAR")["ANNUAL"].mean()

plt.figure(figsize=(12, 6))
sns.lineplot(x=yearly_avg.index, y=yearly_avg.values, marker='o')
plt.title("Average Annual Rainfall Over the Years in India")
plt.xlabel("Year")
plt.ylabel("Average Annual Rainfall (mm)")
plt.grid(True)
plt.tight_layout()
plt.show()
print()

#---Objective 2: Correlation Between Monthly Rainfall---#
monthly_cols = ['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 
                'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC']
corr_matrix = df[monthly_cols].corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap="YlGnBu", fmt=".2f")
plt.title("Correlation Matrix of Monthly Rainfall")
plt.show()
print()

#---Objective 3: Rainfall Distribution in each month---#
monthly_data = df[monthly_cols].melt(var_name="Month", value_name="Rainfall")

plt.figure(figsize=(14, 6))
#sns.boxplot(x="Month", y="Rainfall", data=monthly_data, palette="coolwarm")
sns.boxplot(x="Month", y="Rainfall", data=monthly_data, hue="Month", palette="coolwarm", legend=False)
plt.title("Monthly Rainfall Distribution")
plt.grid(True)
plt.show()
print()

#---Objective 4: Top 10 Subdivisions with Highest Annual Rainfall---#
top_subdivisions = df.groupby("SUBDIVISION")["ANNUAL"].mean().sort_values(ascending=False).head(10)

plt.figure(figsize=(12, 6))
sns.barplot(x=top_subdivisions.values, y=top_subdivisions.index,hue = top_subdivisions.index, palette="viridis", legend = False)
plt.title("Top 10 Subdivisions with Highest Average Annual Rainfall")
plt.xlabel("Average Annual Rainfall (mm)")
plt.tight_layout()
plt.show()
print()

#---Objective 5:  Monthly Contribution to Annual Rainfall---#
monthly_total = df[monthly_cols].sum()
plt.figure(figsize=(10, 10))
plt.pie(monthly_total, labels=monthly_cols, autopct='%1.1f%%', startangle=90, colors=sns.color_palette("Set3"))
plt.title("Monthly Contribution to Total Rainfall in India")
plt.axis('equal')
plt.show()


