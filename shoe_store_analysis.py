
# ======================================================
# SHOE STORE SALES ANALYSIS AND PREDICTION SYSTEM
# ======================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

sns.set_style("whitegrid")

print("="*70)
print("SHOE STORE SALES ANALYSIS AND PREDICTION SYSTEM")
print("="*70)

# ======================================================
# LOAD DATASET
# ======================================================

df = pd.read_csv("shoe_store_sales.csv")

print("\nDataset Loaded Successfully")
print("Rows:", df.shape[0])
print("Columns:", df.shape[1])

print("\nFirst 5 Records")
print(df.head())

# ======================================================
# DATA CLEANING
# ======================================================

df.drop_duplicates(inplace=True)

df["Date"] = pd.to_datetime(df["Date"], dayfirst=True)

df["Month"] = df["Date"].dt.month
df["Month_Name"] = df["Date"].dt.month_name()

print("\nData Cleaning Completed")

# ======================================================
# DESCRIPTIVE STATISTICS
# ======================================================

print("\nDescriptive Statistics")
print(df.describe())

numeric_cols = ["Price", "Quantity", "Discount", "Total_Sales"]

# ======================================================
# UNIVARIATE ANALYSIS
# ======================================================

print("\nUnivariate Analysis")

for col in numeric_cols:

    plt.figure(figsize=(6,4))
    sns.histplot(df[col], kde=True)
    plt.title(f"Distribution of {col}")
    plt.show()

    # BOX PLOT
    plt.figure(figsize=(5,4))
    sns.boxplot(x=df[col])
    plt.title(f"Boxplot of {col}")
    plt.show()

# ======================================================
# BIVARIATE ANALYSIS
# ======================================================

print("\nBivariate Analysis")

sns.scatterplot(data=df, x="Price", y="Total_Sales")
plt.title("Price vs Total Sales")
plt.show()

sns.scatterplot(data=df, x="Quantity", y="Total_Sales")
plt.title("Quantity vs Total Sales")
plt.show()

sns.scatterplot(data=df, x="Discount", y="Total_Sales")
plt.title("Discount vs Total Sales")
plt.show()

# ======================================================
# MULTIVARIATE ANALYSIS
# ======================================================

print("\nMultivariate Analysis")

sns.pairplot(df[numeric_cols])
plt.show()

# ======================================================
# CORRELATION MATRIX
# ======================================================

print("\nCorrelation Matrix")

corr = df[numeric_cols].corr()

print(corr)

plt.figure(figsize=(6,5))
sns.heatmap(corr, annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

# ======================================================
# BRAND SALES ANALYSIS
# ======================================================

brand_sales = df.groupby("Shoe_Brand")["Total_Sales"].sum()

plt.figure(figsize=(8,5))
brand_sales.plot(kind="bar")
plt.title("Brand-wise Sales")
plt.ylabel("Total Sales")
plt.show()

# ======================================================
# MONTHLY SALES ANALYSIS
# ======================================================

monthly_sales = df.groupby("Month_Name")["Total_Sales"].sum()

plt.figure(figsize=(8,5))
monthly_sales.plot(kind="bar")
plt.title("Monthly Sales")
plt.ylabel("Total Sales")
plt.show()

# ======================================================
# MACHINE LEARNING MODEL
# ======================================================

print("\nTraining Linear Regression Model")

X = df[["Price","Quantity","Discount"]]
y = df["Total_Sales"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("R2 Score :", round(r2_score(y_test, y_pred),3))
print("RMSE :", round(np.sqrt(mean_squared_error(y_test, y_pred)),2))

# ======================================================
# SEASONAL SALES PREDICTION
# ======================================================

monthly = df.groupby("Month")["Total_Sales"].sum().reset_index()

season_model = LinearRegression()
season_model.fit(monthly[["Month"]], monthly["Total_Sales"])

monthly["Predicted"] = season_model.predict(monthly[["Month"]])

plt.figure(figsize=(8,5))

plt.plot(monthly["Month"], monthly["Total_Sales"], marker="o", label="Actual")
plt.plot(monthly["Month"], monthly["Predicted"], linestyle="--", label="Predicted")

plt.title("Seasonal Demand Prediction")
plt.xlabel("Month")
plt.ylabel("Sales")
plt.legend()

plt.show()

# ======================================================
# CUSTOMER SEGMENTATION
# ======================================================

print("\nCustomer Segmentation Using K-Means")

cluster_data = df[["Price","Quantity","Total_Sales"]]

scaled = StandardScaler().fit_transform(cluster_data)

kmeans = KMeans(n_clusters=3, random_state=42)

df["Segment"] = kmeans.fit_predict(scaled)

plt.figure(figsize=(7,5))

sns.scatterplot(
    x=df["Price"],
    y=df["Total_Sales"],
    hue=df["Segment"],
    palette="Set1"
)

plt.title("Customer Segmentation")
plt.show()

# ======================================================
# USER INPUT SALES PREDICTION (FIXED WARNING)
# ======================================================

print("\nEnter Data For Sales Prediction")

price = float(input("Enter Shoe Price: "))
quantity = int(input("Enter Quantity Sold: "))
discount = float(input("Enter Discount (%): "))

user_data = pd.DataFrame({
    "Price":[price],
    "Quantity":[quantity],
    "Discount":[discount]
})

prediction = model.predict(user_data)

print("\nPredicted Total Sales:", round(prediction[0],2))

print("\nProject Executed Successfully")
print("="*70)
