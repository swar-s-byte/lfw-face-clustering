import streamlit as st
from sklearn.datasets import fetch_lfw_people
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import numpy as np

# Set page configuration
st.set_page_config(page_title="LFW Face Clustering", layout="wide")

st.title("👤 LFW Face Clustering with PCA + K-Means")

# Load dataset
@st.cache_data
def load_data(min_faces=70, resize=0.4):
    data = fetch_lfw_people(min_faces_per_person=min_faces, resize=resize)
    return data

data = load_data()
X = data.data
images = data.images
names = data.target_names
targets = data.target

st.markdown(f"### Dataset loaded with {X.shape[0]} images and {X.shape[1]} features.")

# Sidebar controls
st.sidebar.title("🔧 Controls")
n_clusters = st.sidebar.slider("Number of Clusters (K)", min_value=2, max_value=20, value=5)
n_components = st.sidebar.slider("PCA Components", min_value=2, max_value=150, value=50)

# PCA Dimensionality Reduction
pca = PCA(n_components=n_components)
X_pca = pca.fit_transform(X)

# Clustering with KMeans
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
labels = kmeans.fit_predict(X_pca)

# Silhouette Score
sil_score = silhouette_score(X_pca, labels)
st.markdown(f"**Silhouette Score**: {sil_score:.3f}")

# Visualize PCA 2D Clusters
st.subheader("📊 Cluster Visualization (First 2 PCA Components)")
fig, ax = plt.subplots(figsize=(8, 6))
scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='tab10', s=30)
legend = ax.legend(*scatter.legend_elements(), title="Clusters")
ax.add_artist(legend)
ax.set_xlabel("PCA Component 1")
ax.set_ylabel("PCA Component 2")
st.pyplot(fig)

# Show faces in selected cluster
st.subheader("🖼️ Browse Faces by Cluster")
selected_cluster = st.selectbox("Select Cluster", list(range(n_clusters)))

cluster_indices = np.where(labels == selected_cluster)[0]
st.markdown(f"**Number of faces in cluster {selected_cluster}: {len(cluster_indices)}**")

# Display image grid
n_cols = 5
for i in range(0, len(cluster_indices), n_cols):
    cols = st.columns(n_cols)
    for j in range(n_cols):
        if i + j < len(cluster_indices):
            idx = cluster_indices[i + j]
            with cols[j]:
                st.image(images[idx], caption=names[targets[idx]], width=100)
