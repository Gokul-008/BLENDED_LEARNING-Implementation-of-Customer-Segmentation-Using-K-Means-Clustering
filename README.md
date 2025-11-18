# BLENDED LEARNING
# Implementation of Customer Segmentation Using K-Means Clustering

## AIM:
To implement customer segmentation using K-Means clustering on the Mall Customers dataset to group customers based on purchasing habits.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Load the Data
Import the dataset to start the clustering analysis process.

2. Explore the Data
Analyze the dataset to understand distributions, patterns, and key characteristics.

3. Select Relevant Features
Identify the most informative features to improve clustering accuracy and relevance.

4. Preprocess the Data
Clean and scale the data to prepare it for clustering.

5. Determine Optimal Number of Clusters
Use techniques like the elbow method to find the ideal number of clusters.

6. Train the Model with K-Means Clustering
Apply the K-Means algorithm to group data points into clusters based on similarity.

7. Analyze and Visualize Clusters
Examine and visualize the resulting clusters to interpret patterns and relationships.

## Program:
```
/*
Program to implement customer segmentation using K-Means clustering on the Mall Customers dataset.
Developed by: GOKUL M
RegisterNumber: 212222230037
*/
# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# Step 1: Data Loading
# Note: The code assumes a file named 'CustomerData.csv' is in the same directory
# The commented out URL in the image was: "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM..."
data = pd.read_csv('CustomerData.csv')

# Step 2: Data Exploration
# Display the first few rows and column names for verification
print(data.head())
print(data.columns)

# Step 3: Feature Selection
# Select relevant features based on the dataset
# Here we will use 'Age', 'Annual Income (k$)', and 'Spending Score (1-100)' for clustering
features = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']
X = data[features]

# Step 4: Data Preprocessing
# Standardize the features to improve K-Means performance
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 5: Determining Optimal Number of Clusters using the Elbow Method
wcss = [] # Within-cluster sum of squares
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, random_state=42)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

# Plot the elbow curve
plt.figure(figsize=(8, 4))
plt.plot(range(1, 11), wcss, marker='o', linestyle='-',color='yellow')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.title('Elbow Method for Optimal Number of Clusters')
plt.show()

# Step 6: Model Training with K-Means Clustering
# Based on the elbow curve, select an appropriate number of clusters, say 4 (adjust as needed based on the plot)
optimal_clusters = 4
kmeans = KMeans(n_clusters=optimal_clusters, random_state=42)
kmeans.fit(X_scaled)

# Step 7: Cluster Analysis and Visualization
# Add the cluster labels to the original dataset
data['Cluster'] = kmeans.labels_

# Calculate and print silhouette score for quality of clustering
sil_score = silhouette_score(X_scaled, kmeans.labels_)
print(f'Silhouette Score: {sil_score}')

# Visualize clusters based on 'Annual Income (k$)' and 'Spending Score (1-100)'
plt.figure(figsize=(10, 6))
sns.scatterplot(data=data, x='Annual Income (k$)', y='Spending Score (1-100)', hue='Cluster', palette='viridis', s=100, alpha=0.7)
plt.title('Customer Segmentation based on Annual Income and Spending Score')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend(title='Cluster')
plt.show()

```

## Output:
<img width="976" height="505" alt="image" src="https://github.com/user-attachments/assets/a88c51dc-1804-487a-84b1-7bab122d7017" />
<img width="1121" height="629" alt="image" src="https://github.com/user-attachments/assets/14f77892-4658-497e-986c-33220ec1698c" />


## Result:
Thus, customer segmentation was successfully implemented using K-Means clustering, grouping customers into distinct segments based on their annual income and spending score. 
