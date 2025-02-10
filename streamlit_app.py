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

# Menyimpan state navigasi jika belum ada
if 'selected_tab' not in st.session_state:
    st.session_state.selected_tab = "Upload Data"

# Buttons for navigation
st.markdown("## 📌 Navigasi")
col1, col2, col3, col4, col5 = st.columns(5)
if col1.button("Upload Data"):
    st.session_state.selected_tab = "Upload Data"
elif col2.button("Visualisasi Data"):
    st.session_state.selected_tab = "Visualisasi Data"
elif col3.button("K-Means Clustering"):
    st.session_state.selected_tab = "K-Means Clustering"
elif col4.button("Random Forest Classification"):
    st.session_state.selected_tab = "Random Forest Classification"
elif col5.button("Input Manual Data"):
    st.session_state.selected_tab = "Input Manual Data"

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
        
        st.subheader("📌 Histogram")
        for col in df_numeric.columns:
            fig = px.histogram(df, x=col, nbins=30, title=f'Distribusi {col}')
            st.plotly_chart(fig, use_container_width=True)
        
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
        
        # Kesimpulan
        st.subheader("Kesimpulan")
        st.write("Dari hasil evaluasi metode, dapat disimpulkan bahwa:")
        
        silhouette_avg = silhouette_score(df[['income', 'score']], df['Cluster'])
        st.write(f"Silhouette Score (K-Means): {silhouette_avg:.2f}")
        
        if accuracy > 0.8:
            st.write("- Random Forest memiliki tingkat akurasi yang tinggi, menunjukkan bahwa metode ini dapat memprediksi dengan baik klasifikasi pelanggan berdasarkan cluster.")
        else:
            st.write("- Akurasi Random Forest masih perlu ditingkatkan dengan optimasi fitur atau parameter model.")
        
        if silhouette_avg > 0.5:
            st.write("- K-Means menunjukkan hasil clustering yang cukup baik berdasarkan nilai silhouette score yang lebih dari 0.5.")
        else:
            st.write("- Hasil clustering dengan K-Means masih bisa ditingkatkan, misalnya dengan mencari jumlah cluster yang lebih optimal.")
        
        st.write("- Secara umum, kombinasi K-Means dan Random Forest dapat digunakan secara efektif untuk segmentasi pelanggan.")
    else:
        st.warning("Silakan jalankan K-Means Clustering terlebih dahulu.")
        
# Menu Input Manual Data
elif menu == "Input Manual Data":
    st.header("✍️ Input Data Manual")
    if 'df' in st.session_state:
        income = st.sidebar.number_input("Masukkan Income", min_value=0, max_value=200, value=50, step=1)
        score = st.sidebar.number_input("Masukkan Spending Score", min_value=0, max_value=100, value=50, step=1)
        
        if st.sidebar.button("Cek Klaster"):
            df = st.session_state.df
            X = df[['income', 'score']]
            kmeans = KMeans(n_clusters=df['Cluster'].nunique(), random_state=42, n_init=10)
            kmeans.fit(X)
            cluster = kmeans.predict([[income, score]])[0]
            st.write(f"Data yang Anda masukkan t ermasuk dalam klaster: {cluster}")
    else:
        st.warning("Silakan upload data terlebih dahulu.")
