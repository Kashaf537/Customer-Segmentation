import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from kneed import KneeLocator
import warnings
warnings.filterwarnings("ignore") 

df = pd.read_csv("D:\\Internship\\Mall_Customers.csv")
print("First 5 rows of the dataset:")
print(df.head())

X = df[['Annual Income (k$)', 'Spending Score (1-100)']]

plt.figure(figsize=(7,5))
plt.scatter(X['Annual Income (k$)'], X['Spending Score (1-100)'], color='blue')
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.title("Original Customer Data")
plt.show()

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

inertia = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)

kneedle = KneeLocator(range(1, 11), inertia, curve="convex", direction="decreasing")
optimal_k = kneedle.elbow
print(f"Optimal number of clusters (K-Means): {optimal_k}")

plt.figure(figsize=(7,5))
plt.plot(range(1, 11), inertia, marker='o')
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Inertia")
plt.title("Elbow Method")
plt.axvline(x=optimal_k, color='red', linestyle='--', label=f'Elbow at k={optimal_k}')
plt.legend()
plt.show()

kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
df['KMeans_Cluster'] = kmeans.fit_predict(X_scaled)

plt.figure(figsize=(7,5))
plt.scatter(df['Annual Income (k$)'], df['Spending Score (1-100)'],
            c=df['KMeans_Cluster'], cmap='viridis', s=50)
centers = scaler.inverse_transform(kmeans.cluster_centers_)
plt.scatter(centers[:,0], centers[:,1], c='red', s=200, marker='X', alpha=0.5, label='Centroids')
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.title(f"K-Means Clusters (k={optimal_k})")
plt.legend()
plt.show()

dbscan = DBSCAN(eps=1.2, min_samples=5)
df['DBSCAN_Cluster'] = dbscan.fit_predict(X_scaled)

plt.figure(figsize=(7,5))
noise = df[df['DBSCAN_Cluster'] == -1]
clusters = df[df['DBSCAN_Cluster'] != -1]

plt.scatter(clusters['Annual Income (k$)'], clusters['Spending Score (1-100)'],
            c=clusters['DBSCAN_Cluster'], cmap='tab10', s=50, label='Clustered')
plt.scatter(noise['Annual Income (k$)'], noise['Spending Score (1-100)'],
            c='red', s=50, label='Noise', marker='x')
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.title("DBSCAN Clusters (Noise in Red)")
plt.legend()
plt.show()

agglo = AgglomerativeClustering(n_clusters=optimal_k)
df['Agglomerative_Cluster'] = agglo.fit_predict(X_scaled)

plt.figure(figsize=(7,5))
markers = ['o','s','^','D','v','P','*','X','H','+']  
for cluster in range(optimal_k):
    subset = df[df['Agglomerative_Cluster']==cluster]
    plt.scatter(subset['Annual Income (k$)'], subset['Spending Score (1-100)'],
                s=50, label=f'Cluster {cluster}', marker=markers[cluster % len(markers)])
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.title(f"Agglomerative Clusters (k={optimal_k})")
plt.legend()
plt.show()

print("K-Means Cluster Analysis:")
print(df.groupby('KMeans_Cluster')[['Annual Income (k$)','Spending Score (1-100)']].mean())

print("\nDBSCAN Cluster Analysis:")
print(df.groupby('DBSCAN_Cluster')[['Annual Income (k$)','Spending Score (1-100)']].mean())

print("\nAgglomerative Cluster Analysis:")
print(df.groupby('Agglomerative_Cluster')[['Annual Income (k$)','Spending Score (1-100)']].mean())

print("\nSample of clustered data:")
print(df.head())