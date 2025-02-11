import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, silhouette_score, accuracy_score

# Set page config
st.set_page_config(page_title="Segmentasi Pelanggan Toserba", page_icon="📊", layout="wide")

# Fungsi untuk membaca data
def load_data(file):
    df = pd.read_csv(file)
    df.columns = df.columns.str.strip()
    df.rename(columns={'spending_score': 'score', 'Annual Income (k$)': 'income'}, inplace=True)
    return df

# Tambahkan CSS untuk menyesuaikan ukuran halaman dan tampilan navbar
st.markdown(
    """
    <style>
    .block-container { 
        padding-top: 1rem;
        max-width: 1100px; 
    }
    .stTabs [role="tablist"] { 
        justify-content: center;
        margin-top: 50px;  /* Menurunkan posisi navbar */
    }
    .stTabs [role="tab"] { 
        font-size: 35px; 
        font-weight: bold; 
        padding: 15px 25px; 
        border-radius: 8px;
    }
    .stTabs [role="tab"]:hover { 
        color: white; 
        background-color: #007bff; 
    }
    .stTabs [role="tab"][aria-selected="true"] { 
        color: white; 
        background-color: #007bff; 
        font-size: 24px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Sidebar untuk upload data
st.sidebar.header("📂 Upload Data")
uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=['csv'])

# Data sampel
sample_data = pd.DataFrame({
    'income': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
    'score': [15, 25, 35, 45, 55, 65, 75, 85, 95, 5]
})

# Memuat data
if uploaded_file:
    df = load_data(uploaded_file)
    st.sidebar.success("Data berhasil diunggah!")
else:
    try:
        df = load_data("data_customer.csv")
        st.sidebar.info("Menggunakan data bawaan: data_customer.csv")
    except FileNotFoundError:
        df = sample_data
        st.sidebar.warning("Menampilkan data sampel. Silakan upload file untuk menggantinya.")

# Validasi kolom
required_columns = {'income', 'score'}
if not required_columns.issubset(df.columns):
    st.error(f"Dataset harus memiliki kolom: {', '.join(required_columns)}")
    st.stop()

X = df[['income', 'score']]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Tabs sebagai navbar
tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Visualisasi Data", "📈 K-Means", "🌲 Random Forest", "📋 Dashboard", "📊 Perbandingan Metode"])

# ---- Tab 1: Visualisasi Data ----
with tab1:
    st.header("📊 Visualisasi Data")
    col1, col2 = st.columns([2, 1])
    
    with col1:
        fig = px.line(df, x=df.index, y='income', title="📈 Tren Income Pelanggan")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.pie(df, names="score", title="🎯 Distribusi Spending Score", hole=0.4)
        st.plotly_chart(fig, use_container_width=True)
    # Menambahkan pie chart untuk distribusi spending score
spending_bins = pd.cut(df['score'], bins=[0, 20, 40, 60, 80, 100], labels=["0-20", "21-40", "41-60", "61-80", "81-100"])
spending_counts = spending_bins.value_counts().reset_index()
spending_counts.columns = ['Range', 'Count']
fig_pie = px.pie(spending_counts, names='Range', values='Count', title="🔵 Distribusi Spending Score")
st.plotly_chart(fig_pie, use_container_width=True)


# ---- Tab 2: K-Means Clustering ----
with tab2:
    st.header("📈 K-Means Clustering")
    st.markdown("### Evaluasi dengan Elbow Method")
    
    @st.cache_data
    def calculate_inertia(X_scaled, max_k=10):
        inertia = []
        for k in range(1, max_k + 1):
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(X_scaled)
            inertia.append(kmeans.inertia_)
        return inertia
    
    inertia = calculate_inertia(X_scaled)
    fig = px.line(x=range(1, 11), y=inertia, markers=True, title="Elbow Method untuk Menentukan Jumlah Cluster")
    fig.update_layout(xaxis_title="Jumlah Cluster", yaxis_title="Inertia")
    st.plotly_chart(fig, use_container_width=True)
    
    num_clusters = st.slider("Pilih jumlah cluster:", 2, 10, value=3)
    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
    df['Cluster'] = kmeans.fit_predict(X_scaled)
    
    fig = px.scatter(df, x='income', y='score', color=df['Cluster'].astype(str), title="K-Means Clustering", labels={'color': 'Cluster'})
    st.plotly_chart(fig, use_container_width=True)

# ---- Tab 3: Random Forest ----
with tab3:
    st.header("🌲 Random Forest Classification")
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, df['Cluster'], test_size=0.3, random_state=42)
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    
    st.subheader("Classification Report")
    st.text(classification_report(y_test, y_pred))
    
    cm = confusion_matrix(y_test, y_pred)
    fig = px.imshow(cm, text_auto=True, color_continuous_scale='Blues', title="Confusion Matrix")
    st.plotly_chart(fig, use_container_width=True)

# ---- Tab 4: Dashboard ----
with tab4:
    st.header("📋 Dashboard Segmentasi Pelanggan")
    cluster_filter = st.multiselect("Pilih Cluster untuk ditampilkan:", options=df['Cluster'].unique(), default=df['Cluster'].unique())
    filtered_df = df[df['Cluster'].isin(cluster_filter)]
    
    col1, col2 = st.columns([2, 1])
    with col1:
        fig = px.scatter(filtered_df, x='income', y='score', color=filtered_df['Cluster'].astype(str), title="Scatter Plot Filtered by Cluster")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        cluster_counts = filtered_df['Cluster'].value_counts()
        fig = px.pie(cluster_counts, names=cluster_counts.index, title="Distribusi Klaster Pelanggan")
        st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("📋 Hasil Segmentasi Pelanggan")
    st.dataframe(filtered_df.head(20))
    
    st.subheader("📊 Distribusi Data Pelanggan")
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.boxplot(data=filtered_df[['income', 'score']], ax=ax)
    st.pyplot(fig)

# ---- Metrik Penting ----
st.sidebar.header("📊 Metrik Penting")
st.sidebar.metric("Total Pelanggan", df.shape[0])
st.sidebar.metric("Jumlah Klaster", df['Cluster'].nunique())
st.sidebar.metric("Akurasi Random Forest", f"{accuracy_score(y_test, y_pred) * 100:.2f}%")


# ---- Tab 5: Perbandingan Metode ----
with tab5:
    st.header("📊 Perbandingan Metode K-Means vs Random Forest")
    st.write("Di sini kita bisa membandingkan performa K-Means dan Random Forest.")
    st.subheader("Silhouette Score untuk K-Means")
    fig_silhouette = px.line(x=range(3, 12, 2), y=silhouette_scores, markers=True, title="Silhouette Score")
    fig_silhouette.update_layout(xaxis_title="Jumlah Cluster", yaxis_title="Silhouette Score")
    st.plotly_chart(fig_silhouette, use_container_width=True)
    
    st.subheader("Akurasi Random Forest")
    st.metric("Akurasi", f"{accuracy_score(y_test, y_pred) * 100:.2f}%")
