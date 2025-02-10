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

# Tabs for navigation
tabs = ["Upload Data", "Visualisasi Data", "K-Means Clustering", "Random Forest Classification", "Input Manual Data"]
selected_tab = st.selectbox("Pilih Halaman:", tabs)

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
        
        # Filter hanya kolom numerik
        df_numeric = df.select_dtypes(include=[np.number])
        
        st.subheader("📌 Correlation Heatmap")
        if df_numeric.empty:
            st.warning("Tidak ada kolom numerik dalam dataset.")
        else:
            corr_matrix = df_numeric.corr()
            fig = px.imshow(corr_matrix, text_auto=True, color_continuous_scale='RdBu', title="Correlation Heatmap")
            st.plotly_chart(fig, use_container_width=True)
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
            
# K-Means Clustering Section
elif selected_tab == "K-Means Clustering":
    st.header("📈 K-Means Clustering")
    if 'df' not in st.session_state:
        st.warning("Silakan upload data terlebih dahulu.")
    else:
        X_scaled = st.session_state.X_scaled
        
        st.subheader("Elbow Method")
        k_range = range(2, 11)
        inertia = []
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(X_scaled)
            inertia.append(kmeans.inertia_)
    st.subheader("Silhouette Score")
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(k_range, silhouette, marker='o', linestyle='-', color='green')
        ax.set_xlabel("Number of Clusters (K)")
        ax.set_ylabel("Silhouette Score")
        ax.set_title("Silhouette Score")
        st.pyplot(fig)
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
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=list(k_range), y=inertia, mode='lines+markers', marker=dict(color='blue')))
        fig.update_layout(title="Elbow Method", xaxis_title="Number of Clusters (K)", yaxis_title="Inertia")
        st.plotly_chart(fig, use_container_width=True)
        
        n_clusters = st.slider("Pilih Jumlah Cluster:", min_value=2, max_value=10, value=3, step=1)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        df = st.session_state.df.copy()
        df['Cluster'] = kmeans.fit_predict(X_scaled)
        st.session_state.df = df
        
        st.subheader("Visualisasi Hasil K-Means")
        fig = px.scatter(df, x='income', y='score', color='Cluster', title="K-Means Clustering", color_continuous_scale='viridis')
        st.plotly_chart(fig, use_container_width=True)

# Random Forest Classification Section
elif selected_tab == "Random Forest Classification":
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
        
        cm = confusion_matrix(y_test, y_pred)
        fig = px.imshow(cm, text_auto=True, color_continuous_scale='Blues', title="Confusion Matrix")
        st.plotly_chart(fig, use_container_width=True)

# Perbandingan Metode K-Means vs Random Forest
elif menu == "Perbandingan Metode":
    st.header("📊 Perbandingan Metode K-Means vs Random Forest")
    if 'df' in st.session_state and 'Cluster' in st.session_state.df.columns:
        df = st.session_state.df
        X_train, X_test, y_train, y_test = train_test_split(df[['income', 'score']], df['Cluster'], test_size=0.3, random_state=42)
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_test)
        
        st.subheader("Evaluasi Metode")
        accuracy = accuracy_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        st.write(f"Accuracy Score (Random Forest): {accuracy:.2f}")
        st.write(f"Mean Absolute Error (Random Forest): {mae:.2f}")
        st.write(f"Mean Squared Error (Random Forest): {mse:.2f}")
        
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_title("Confusion Matrix")
        st.pyplot(fig)

# Input Manual Data Section
elif selected_tab == "Input Manual Data":
    st.header("✍️ Input Data Manual")
    if 'df' in st.session_state and 'Cluster' in st.session_state.df.columns:
        col1, col2 = st.columns(2)
        with col1:
            income = st.number_input("💰 Masukkan Income (dalam ribuan $)", min_value=0, max_value=200, value=50, step=1)
        with col2:
            score = st.number_input("📈 Masukkan Spending Score (0-100)", min_value=0, max_value=100, value=50, step=1)
        
        if st.button("Cek Klaster"):
            df = st.session_state.df
            X = df[['income', 'score']]
            kmeans = KMeans(n_clusters=df['Cluster'].nunique(), random_state=42, n_init=10)
            kmeans.fit(X)
            cluster = kmeans.predict([[income, score]])[0]
            st.success(f"Data yang Anda masukkan termasuk dalam klaster: **{cluster}**")
