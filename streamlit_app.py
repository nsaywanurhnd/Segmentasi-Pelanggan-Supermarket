import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, silhouette_score, mean_absolute_error, mean_squared_error, accuracy_score
from sklearn.decomposition import PCA

# Set page config
st.set_page_config(page_title="Segmentasi Pelanggan Toserba", page_icon="📊", layout="wide")
st.title("📊 Segmentasi Pelanggan Toserba")

# Sidebar menu
st.sidebar.title("Navigasi")
menu = st.sidebar.radio("Pilih Menu", ["Upload Data", "Visualisasi Data", "K-Means Clustering", "Random Forest Classification", "Perbandingan Metode", "Input Manual Data"])

# Upload Data Section
if menu == "Upload Data":
    st.header("📂 Upload Data")
    st.write("Segmentasi pelanggan membantu bisnis dalam memahami kelompok pelanggan berdasarkan pola belanja mereka, sehingga dapat meningkatkan strategi pemasaran dan layanan.")
    
    uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=['csv'])
    
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
    
    # Fitur Pencarian Data
    st.subheader("🔍 Pencarian Data")
    search_column = st.selectbox("Pilih kolom untuk pencarian:", df.columns)
    search_value = st.text_input("Masukkan nilai untuk pencarian:")
    if search_value:
        filtered_df = df[df[search_column].astype(str).str.contains(search_value, case=False, na=False)]
        st.write(f"Menampilkan hasil pencarian untuk '{search_value}' di kolom '{search_column}':")
        st.dataframe(filtered_df, use_container_width=True)
    
    # Data cleaning and preparation
    df.columns = df.columns.str.strip()
    df.rename(columns={'spending_score': 'score', 'Annual Income (k$)': 'income'}, inplace=True)
    
    st.session_state.df = df
    X = df[['income', 'score']]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    st.session_state.X_scaled = X_scaled

# Visualisasi Data Section
elif menu == "Visualisasi Data":
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

# K-Means Clustering Section
elif menu == "K-Means Clustering":
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
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=list(k_range), y=inertia, mode='lines+markers', marker=dict(color='blue')))
        fig.update_layout(title="Elbow Method", xaxis_title="Number of Clusters (K)", yaxis_title="Inertia")
        st.plotly_chart(fig, use_container_width=True)
        
        n_clusters = st.sidebar.slider("Pilih Jumlah Cluster:", min_value=2, max_value=10, value=3, step=1)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        df = st.session_state.df.copy()
        df['Cluster'] = kmeans.fit_predict(X_scaled)
        st.session_state.df = df
        
        st.subheader("Visualisasi Hasil K-Means")
        fig = px.scatter(df, x='income', y='score', color='Cluster', title="K-Means Clustering", color_continuous_scale='viridis')
        st.plotly_chart(fig, use_container_width=True)

# Random Forest Classification Section
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
        
        cm = confusion_matrix(y_test, y_pred)
        fig = px.imshow(cm, text_auto=True, color_continuous_scale='Blues', title="Confusion Matrix")
        st.plotly_chart(fig, use_container_width=True)

# Perbandingan Metode Section
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
        
        cm = confusion_matrix(y_test, y_pred)
        fig = px.imshow(cm, text_auto=True, color_continuous_scale='Blues', title="Confusion Matrix")
        st.plotly_chart(fig, use_container_width=True)
        
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

# Input Manual Data Section
elif menu == "Input Manual Data":
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
            
            if 'kmeans_model' not in st.session_state:
                kmeans = KMeans(n_clusters=df['Cluster'].nunique(), random_state=42, n_init=10)
                kmeans.fit(X)
                st.session_state.kmeans_model = kmeans
            else:
                kmeans = st.session_state.kmeans_model
            
            cluster = kmeans.predict([[income, score]])[0]
            st.success(f"Data yang Anda masukkan termasuk dalam klaster: **{cluster}**")
            
            st.subheader("📊 Visualisasi Klaster dengan Data Input Manual")
            fig = px.scatter(df, x='income', y='score', color='Cluster', title="K-Means Clustering dengan Data Baru", color_continuous_scale='viridis')
            fig.add_trace(go.Scatter(x=[income], y=[score], mode='markers', marker=dict(size=12, color='red'), name='Input Manual'))
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Silakan upload data dan jalankan K-Means Clustering terlebih dahulu.")
