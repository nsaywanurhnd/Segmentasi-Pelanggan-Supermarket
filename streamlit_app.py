import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, silhouette_score, accuracy_score

# Set page config
st.set_page_config(page_title="Segmentasi Pelanggan Toserba", page_icon="📊", layout="wide")
st.title("📊 Segmentasi Pelanggan Toserba")
st.markdown("**Tujuan Website:** Menganalisis dan mengelompokkan pelanggan berdasarkan pola pembelian mereka menggunakan K-Means dan Random Forest.")

# Upload Data
st.header("📂 Upload Data")
uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success("Data berhasil diunggah!")
else:
    try:
        df = pd.read_csv("data_customer.csv")
        st.info("Menggunakan data bawaan: data_customer.csv")
    except FileNotFoundError:
        st.error("File data_customer.csv tidak ditemukan. Silakan upload file terlebih dahulu.")
        st.stop()

st.write("Data yang digunakan:")
st.dataframe(df, use_container_width=True)

df.columns = df.columns.str.strip()
df.rename(columns={'spending_score': 'score', 'Annual Income (k$)': 'income'}, inplace=True)

X = df[['income', 'score']]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# K-Means Clustering
st.header("📈 K-Means Clustering")
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
df['Cluster'] = kmeans.fit_predict(X_scaled)

st.subheader("Visualisasi Hasil K-Means")
fig, ax = plt.subplots(figsize=(10, 6))
scatter = ax.scatter(df['income'], df['score'], c=df['Cluster'], cmap='viridis')
ax.set_xlabel("Income")
ax.set_ylabel("Spending Score")
ax.set_title("K-Means Clustering")
st.pyplot(fig)

# Random Forest Classification
st.header("🌲 Random Forest Classification")
X_train, X_test, y_train, y_test = train_test_split(df[['income', 'score']], df['Cluster'], test_size=0.3, random_state=42)
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)

st.subheader("Classification Report")
st.text(classification_report(y_test, y_pred))

fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues', ax=ax)
ax.set_title("Confusion Matrix")
st.pyplot(fig)

# Perbandingan K-Means dan Random Forest
st.header("📊 Perbandingan Metode K-Means vs Random Forest")
accuracy = accuracy_score(y_test, y_pred)
silhouette_avg = silhouette_score(df[['income', 'score']], df['Cluster'])

st.write(f"Silhouette Score (K-Means): {silhouette_avg:.2f}")
st.write(f"Accuracy Score (Random Forest): {accuracy:.2f}")
st.write("Kesimpulan: K-Means lebih baik untuk segmentasi, sedangkan Random Forest lebih baik untuk prediksi berdasarkan cluster.")
