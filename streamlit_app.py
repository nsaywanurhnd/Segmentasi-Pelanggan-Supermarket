import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# Set page config
st.set_page_config(page_title="Segmentasi Pelanggan Toserba", page_icon="📊", layout="wide")
st.title("📊 Segmentasi Pelanggan Toserba")
st.markdown("**Tujuan Website:** Menganalisis dan mengelompokkan pelanggan berdasarkan pola pembelian mereka menggunakan K-Means dan Random Forest.")

# Buttons for navigation
st.markdown("## 📌 Navigasi")
col1, col2, col3, col4, col5 = st.columns(5)
if col1.button("Upload Data"):
    selected_tab = "Upload Data"
elif col2.button("Visualisasi Data"):
    selected_tab = "Visualisasi Data"
elif col3.button("K-Means Clustering"):
    selected_tab = "K-Means Clustering"
elif col4.button("Random Forest Classification"):
    selected_tab = "Random Forest Classification"
elif col5.button("Input Manual Data"):
    selected_tab = "Input Manual Data"
else:
    selected_tab = "Upload Data"

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

# Visualisasi Data Section
elif selected_tab == "Visualisasi Data":
    st.header("📊 Visualisasi Data")
    if 'df' not in st.session_state:
        st.warning("Silakan upload data terlebih dahulu.")
    else:
        df = st.session_state.df
        df_numeric = df.select_dtypes(include=[np.number])
        st.subheader("📌 Correlation Heatmap")
        if df_numeric.empty:
            st.warning("Tidak ada kolom numerik dalam dataset.")
        else:
            corr_matrix = df_numeric.corr()
            fig = px.imshow(corr_matrix, text_auto=True, color_continuous_scale='RdBu', title="Correlation Heatmap")
            st.plotly_chart(fig, use_container_width=True)
    numerical_features = [col for col in ['age', 'income', 'score', 'membership_years', 'purchase_frequency', 'last_purchase_amount'] if col in df.columns]
        
        st.subheader("📌 Histogram")
        fig, axes = plt.subplots(len(numerical_features) // 2, 2, figsize=(15, 10))
        for i, feature in enumerate(numerical_features):
            ax = axes[i // 2, i % 2]
            df[feature].hist(bins=15, ax=ax, grid=False)
            ax.set_title(f'Distribution of {feature.capitalize()}')
            ax.set_xlabel(feature.capitalize())
            ax.set_ylabel('Frequency')
        plt.tight_layout()
        st.pyplot(fig)
        
        categorial_features = [col for col in ['gender', 'preferred_category'] if col in df.columns]
        if categorial_features:
            st.subheader("📌 Kategori Data")
            fig, axes = plt.subplots(1, len(categorial_features), figsize=(15,6))
            for i, feature in enumerate(categorial_features):
                sns.countplot(x=feature, data=df, ax=axes[i], palette='terrain')
                axes[i].set_title(f'Distribution of {feature.capitalize()}')
                axes[i].set_xlabel(feature.capitalize())
            plt.tight_layout()
            st.pyplot(fig)
        
        if 'income' in df.columns:
            st.subheader("📌 Density Plot for Income")
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.kdeplot(df['income'], color="yellow", shade=True)
            plt.title("Density Plot for Annual Income")
            st.pyplot(fig)
        

elif menu == "K-Means Clustering":
    st.header("📈 K-Means Clustering")
    if 'df' not in st.session_state:
        st.warning("Silakan upload data terlebih dahulu.")
    else:
        df = st.session_state.df
        X_scaled = st.session_state.X_scaled
        
        k_range = range(2, 11)
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
        
        n_clusters = st.sidebar.slider("Pilih Jumlah Cluster:", min_value=2, max_value=10, value=3, step=1)
        if st.sidebar.button("Run Clustering"):
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            df['Cluster'] = kmeans.fit_predict(X_scaled)
            
            st.subheader("Visualisasi Hasil K-Means")
            fig, ax = plt.subplots(figsize=(10, 6))
            scatter = ax.scatter(df['income'], df['score'], c=df['Cluster'], cmap='viridis')
            ax.set_xlabel("Income")
            ax.set_ylabel("Spending Score")
            ax.set_title("K-Means Clustering")
            st.pyplot(fig)
             
            st.subheader("Deskripsi Tiap Cluster")
            for i in range(n_clusters):
                cluster_data = df[df['Cluster'] == i]
                st.write(f"**Cluster {i}:**")
                st.write(f"- Rata-rata Income: ${cluster_data['income'].mean():,.2f}")
                st.write(f"- Rata-rata Spending Score: {cluster_data['score'].mean():.2f}")
                st.write(f"- Jumlah Anggota: {len(cluster_data)} orang")
            st.session_state.df = df
           

elif menu == "Random Forest Classification":
    st.header("🌲 Random Forest Classification")
    if 'df' not in st.session_state or 'Cluster' not in st.session_state.df.columns:
        st.warning("Silakan jalankan K-Means Clustering terlebih dahulu.")
    else:
        df = st.session_state.df
        X_train, X_test, y_train, y_test = train_test_split(st.session_state.X_scaled, df['Cluster'], test_size=0.3, random_state=42)
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_test)
        
        st.subheader("Classification Report")
        st.text(classification_report(y_test, y_pred))
        
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_title("Confusion Matrix")
        st.pyplot(fig)


