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
from sklearn.metrics import classification_report, confusion_matrix, silhouette_score, accuracy_score, mean_absolute_error, mean_squared_error

# Set page config
st.set_page_config(page_title="Segmentasi Pelanggan Toserba", page_icon="📊", layout="wide")
st.title("📊 Segmentasi Pelanggan Toserba")
st.markdown("**Tujuan Website:** Menganalisis dan mengelompokkan pelanggan berdasarkan pola pembelian mereka menggunakan K-Means dan Random Forest.")


# Load data
@st.cache_data
def load_data():
    return pd.read_csv("data_customer.csv")

df = load_data()
st.sidebar.header("📂 Upload Data")
uploaded_file = st.sidebar.file_uploader("Upload CSV", type=['csv'])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
st.sidebar.success("Data berhasil diunggah!")

# Preprocessing
df.columns = df.columns.str.strip()
df.rename(columns={'spending_score': 'score', 'Annual Income (k$)': 'income'}, inplace=True)
X = df[['income', 'score']]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# K-Means Clustering
kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
df['Cluster'] = kmeans.fit_predict(X_scaled)

# Random Forest Classification
X_train, X_test, y_train, y_test = train_test_split(X, df['Cluster'], test_size=0.3, random_state=42)
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# Metrics
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Users", len(df))
col2.metric("Unique Clusters", len(df['Cluster'].unique()))
col3.metric("Model Accuracy", f"{accuracy:.2%}")
col4.metric("Silhouette Score", f"{silhouette_score(X, df['Cluster']):.2f}")

# Line Chart
st.subheader("📈 Session Trends")
time_series = np.random.randint(10, 50, size=30)  # Simulated session data
dates = pd.date_range(start="2023-01-01", periods=30)
fig = px.line(x=dates, y=time_series, labels={'x': 'Date', 'y': 'Sessions'})
st.plotly_chart(fig, use_container_width=True)

# Pie Chart - Cluster Distribution
st.subheader("🔄 Cluster Distribution")
fig = px.pie(df, names='Cluster', title="Percentage of Customers in Each Cluster")
st.plotly_chart(fig, use_container_width=True)

# Confusion Matrix
st.subheader("🎯 Confusion Matrix")
fig, ax = plt.subplots(figsize=(6, 4))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues', ax=ax)
st.pyplot(fig)

# Menyimpan state navigasi jika belum ada
# Set halaman utama
st.set_page_config(page_title="Segmentasi Pelanggan", page_icon="📊", layout="wide")

# Simpan navigasi di session state
if 'selected_tab' not in st.session_state:
    st.session_state.selected_tab = "Upload Data"

# Tombol Navigasi
st.markdown("## 📌 Navigasi")
col1, col2, col3, col4, col5, col6 = st.columns(6)

if col1.button("Upload Data"):
    st.session_state.selected_tab = "Upload Data"
if col2.button("Visualisasi Data"):
    st.session_state.selected_tab = "Visualisasi Data"
if col3.button("K-Means Clustering"):
    st.session_state.selected_tab = "K-Means Clustering"
if col4.button("Random Forest Classification"):
    st.session_state.selected_tab = "Random Forest Classification"
if col5.button("Perbandingan Metode"):
    st.session_state.selected_tab = "Perbandingan Metode"
if col6.button("Input Manual Data"):
    st.session_state.selected_tab = "Input Manual Data"

# Pastikan navigasi tetap ada di setiap halaman
selected_tab = st.session_state.selected_tab

# Upload Data Section
if selected_tab == "Upload Data":
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
    
    # Data cleaning and preparation
    df.columns = df.columns.str.strip()
    df.rename(columns={'spending_score': 'score', 'Annual Income (k$)': 'income'}, inplace=True)
    
    st.session_state.df = df
    X = df[['income', 'score']]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    st.session_state.X_scaled = X_scaled

# K-Means Clustering Section
elif selected_tab == "K-Means Clustering":
    st.header("📈 K-Means Clustering")
    if 'df' not in st.session_state:
        st.warning("Silakan upload data terlebih dahulu.")
    else:
        df = st.session_state.df
        X_scaled = st.session_state.X_scaled
        
        k_range = range(3, 12, 2)  # Menggunakan angka ganjil mulai dari 3
        inertia, silhouette = [], []
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(X_scaled)
            inertia.append(kmeans.inertia_)
            silhouette.append(silhouette_score(X_scaled, labels))
        
        st.subheader("Elbow Method")
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(k_range, inertia, marker='o', linestyle='-', color='blue')
        ax.set_xlabel("Number of Clusters (K)")
        ax.set_ylabel("Inertia")
        ax.set_title("Elbow Method")
        st.pyplot(fig)
        
        st.subheader("Silhouette Score")
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(k_range, silhouette, marker='o', linestyle='-', color='green')
        ax.set_xlabel("Number of Clusters (K)")
        ax.set_ylabel("Silhouette Score")
        ax.set_title("Silhouette Score")
        st.pyplot(fig)
        
        n_clusters = st.sidebar.slider("Pilih Jumlah Cluster:", min_value=3, max_value=11, value=3, step=2)
        if st.sidebar.button("Run Clustering"):
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            df['Cluster'] = kmeans.fit_predict(X_scaled)
            st.session_state.df = df
            
            st.subheader("Visualisasi Hasil K-Means")
            fig, ax = plt.subplots(figsize=(10, 6))
            scatter = ax.scatter(df['income'], df['score'], c=df['Cluster'], cmap='viridis')
            ax.set_xlabel("Income")
            ax.set_ylabel("Spending Score")
            ax.set_title("K-Means Clustering")
            st.pyplot(fig)
            
# Random Forest Classification
elif selected_tab == "Random Forest Classification":
    st.header("🌲 Random Forest Classification")
    if 'df' not in st.session_state or 'Cluster' not in st.session_state.df.columns:
        st.warning("Silakan jalankan K-Means Clustering terlebih dahulu.")
    else:
        df = st.session_state.df
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
elif selected_tab == "Perbandingan Metode":
    st.header("📊 Perbandingan Metode K-Means vs Random Forest")
    if 'df' in st.session_state and 'Cluster' in st.session_state.df.columns:
        df = st.session_state.df
        X_train, X_test, y_train, y_test = train_test_split(df[['income', 'score']], df['Cluster'], test_size=0.3, random_state=42)
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_test)
        
        st.subheader("Evaluasi Metode")
        accuracy = accuracy_score(y_test, y_pred)
        silhouette_avg = silhouette_score(df[['income', 'score']], df['Cluster'])
        st.write(f"Silhouette Score (K-Means): {silhouette_avg:.2f}")
        st.write(f"Accuracy Score (Random Forest): {accuracy:.2f}")
        st.write("Kesimpulan: K-Means lebih baik untuk segmentasi, sedangkan Random Forest lebih baik untuk prediksi berdasarkan cluster.")
