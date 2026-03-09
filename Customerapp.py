import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering

# Page config
st.set_page_config(page_title="Customer Segmentation Dashboard", layout="wide")

st.title("🛍 Customer Segmentation Dashboard")
st.write("Clustering using **KMeans, DBSCAN, and Agglomerative Clustering**")

# Sidebar controls
st.sidebar.header("Controls")

uploaded_file = st.sidebar.file_uploader("Upload Mall Customers CSV", type=["csv"])

# Load dataset
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
else:
    st.info("Using default dataset (Mall_Customers.csv)")
    try:
        df = pd.read_csv("D:/Arch/Mall_Customers.csv")
    except:
        st.warning("Please upload Mall_Customers.csv")
        st.stop()

# Dataset preview
st.subheader("Dataset Preview")
st.dataframe(df.head(), use_container_width=True)

# Check required columns
required_cols = ['Annual Income (k$)', 'Spending Score (1-100)']

if not all(col in df.columns for col in required_cols):
    st.error("Dataset must contain columns: 'Annual Income (k$)' and 'Spending Score (1-100)'")
    st.stop()

income_col = 'Annual Income (k$)'
spending_col = 'Spending Score (1-100)'

X = df[[income_col, spending_col]]

# Plot original data
st.subheader("Customer Distribution")

fig, ax = plt.subplots()

ax.scatter(
    X[income_col],
    X[spending_col]
)

ax.set_xlabel("Annual Income (k$)")
ax.set_ylabel("Spending Score (1-100)")
ax.set_title("Customer Data")

st.pyplot(fig)

# Standardization
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# -----------------------
# KMEANS
# -----------------------

st.subheader("KMeans Clustering")

k = st.sidebar.slider("KMeans Clusters", 2, 10, 5)

kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)

df["KMeans Cluster"] = kmeans.fit_predict(X_scaled)

centers = scaler.inverse_transform(kmeans.cluster_centers_)

fig, ax = plt.subplots()

ax.scatter(
    df[income_col],
    df[spending_col],
    c=df["KMeans Cluster"],
    cmap="viridis"
)

ax.scatter(
    centers[:,0],
    centers[:,1],
    marker="X",
    s=200,
    label="Centroids"
)

ax.set_xlabel("Annual Income")
ax.set_ylabel("Spending Score")
ax.set_title("KMeans Clusters")
ax.legend()

st.pyplot(fig)

# -----------------------
# DBSCAN
# -----------------------

st.subheader("DBSCAN Clustering")

eps = st.sidebar.slider("DBSCAN eps", 0.1, 5.0, 1.0)
min_samples = st.sidebar.slider("DBSCAN min_samples", 2, 10, 5)

dbscan = DBSCAN(eps=eps, min_samples=min_samples)

df["DBSCAN Cluster"] = dbscan.fit_predict(X_scaled)

fig, ax = plt.subplots()

ax.scatter(
    df[income_col],
    df[spending_col],
    c=df["DBSCAN Cluster"],
    cmap="tab10"
)

ax.set_xlabel("Annual Income")
ax.set_ylabel("Spending Score")
ax.set_title("DBSCAN Clusters")

st.pyplot(fig)

# -----------------------
# AGGLOMERATIVE
# -----------------------

st.subheader("Agglomerative Clustering")

agg_clusters = st.sidebar.slider("Agglomerative Clusters", 2, 10, 5)

agg = AgglomerativeClustering(n_clusters=agg_clusters)

df["Agglomerative Cluster"] = agg.fit_predict(X_scaled)

fig, ax = plt.subplots()

ax.scatter(
    df[income_col],
    df[spending_col],
    c=df["Agglomerative Cluster"],
    cmap="rainbow"
)

ax.set_xlabel("Annual Income")
ax.set_ylabel("Spending Score")
ax.set_title("Agglomerative Clusters")

st.pyplot(fig)

# -----------------------
# Cluster Analysis
# -----------------------

st.subheader("Cluster Analysis")

col1, col2, col3 = st.columns(3)

with col1:
    st.write("### KMeans")
    st.dataframe(
        df.groupby("KMeans Cluster")[[income_col, spending_col]].mean()
    )

with col2:
    st.write("### DBSCAN")
    st.dataframe(
        df.groupby("DBSCAN Cluster")[[income_col, spending_col]].mean()
    )

with col3:
    st.write("### Agglomerative")
    st.dataframe(
        df.groupby("Agglomerative Cluster")[[income_col, spending_col]].mean()
    )

# Show clustered dataset
st.subheader("Clustered Dataset")
st.dataframe(df.head(20), use_container_width=True)