import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

st.set_option('deprecation.showPyplotGlobalUse', False)
st.title("🎶 Spotify Clustering Explorer")

# Upload CSV
uploaded_file = st.file_uploader("Upload your SpotifyFeatures.csv", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    features = ['danceability', 'energy', 'loudness', 'speechiness',
                'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']
    X = df[features].dropna()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Choose clustering method
    method = st.radio("Choose clustering method", ['KMeans', 'DBSCAN'])

    if method == 'KMeans':
        k = st.slider("Number of clusters", 2, 10, 4)
        kmeans = KMeans(n_clusters=k, random_state=42)
        labels = kmeans.fit_predict(X_scaled)
    else:
        eps = st.slider("DBSCAN eps", 0.1, 5.0, 1.5)
        min_samples = st.slider("Min samples", 1, 10, 5)
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        labels = dbscan.fit_predict(X_scaled)

    # Evaluation
    if len(set(labels)) > 1 and -1 not in set(labels):
        st.write("Silhouette Score:", silhouette_score(X_scaled, labels))
        st.write("Davies-Bouldin Index:", davies_bouldin_score(X_scaled, labels))
    elif -1 in labels:
        st.warning("DBSCAN detected noise points. Evaluation only includes core clusters.")
        mask = labels != -1
        if np.any(mask):
            st.write("Silhouette Score:", silhouette_score(X_scaled[mask], np.array(labels)[mask]))
            st.write("Davies-Bouldin Index:", davies_bouldin_score(X_scaled[mask], np.array(labels)[mask]))
        else:
            st.error("No core clusters found.")

    # PCA plot
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    df_vis = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])
    df_vis['Cluster'] = labels

    sns.scatterplot(data=df_vis, x='PC1', y='PC2', hue='Cluster', palette='Set2')
    plt.title(f"{method} Clusters")
    st.pyplot()