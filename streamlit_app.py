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
from sklearn.metrics import classification_report, confusion_matrix, silhouette_score, accuracy_score, mean_absolute_error, mean_squared_error

# Set page config
st.set_page_config(page_title="Segmentasi Pelanggan Toserba", page_icon="📊", layout="wide")

# ---- NAVIGASI MENU ----
menu = st.sidebar.radio("Pilih Menu", ["Upload Data", "Visualisasi Data", "K-Means Clustering", "Random Forest Classification", "Perbandingan Metode", "Input Manual Data"])

# ---- UPLOAD DATA ----
if menu == "Upload Data":
    st.header("📂 Upload Data")
    uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.session_state.df = df
        st.success("Data berhasil diunggah!")

    if "df" in st.session_state:
        st.write("**Data yang digunakan:**")
        st.dataframe(st.session_state.df, use_container_width=True)
    else:
        st.warning("Silakan upload file terlebih dahulu.")

# ---- VISUALISASI DATA ----
elif menu == "Visualisasi Data":
    st.header("📊 Visualisasi Data")

    if "df" in st.session_state:
        df = st.session_state.df
        df.rename(columns={'spending_score': 'score', 'Annual Income (k$)': 'income'}, inplace=True)

        col1, col2 = st.columns([2, 1])
        with col1:
            fig = px.line(df, x=df.index, y='income', title="📈 Tren Income Pelanggan")
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            fig = px.pie(df, names="score", title="🎯 Distribusi Spending Score", hole=0.4)
            st.plotly_chart(fig, use_container_width=True)

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Pelanggan", len(df))
        col2.metric("Rata-rata Income", f"${df['income'].mean():,.2f}")
        col3.metric("Rata-rata Spending Score", f"{df['score'].mean():.2f}")
        col4.metric("Max Spending Score", f"{df['score'].max()}")

    else:
        st.warning("Silakan upload data terlebih dahulu.")

# ---- K-MEANS CLUSTERING ----
elif menu == "K-Means Clustering":
    st.header("📈 K-Means Clustering")

    if "df" in st.session_state:
        df = st.session_state.df
        X = df[['income', 'score']]
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
        df['Cluster'] = kmeans.fit_predict(X_scaled)

        st.session_state.df = df
        st.session_state.X_scaled = X_scaled

        fig, ax = plt.subplots(figsize=(10, 6))
        scatter = ax.scatter(df['income'], df['score'], c=df['Cluster'], cmap='viridis')
        ax.set_xlabel("Income")
        ax.set_ylabel("Spending Score")
        ax.set_title("K-Means Clustering")
        st.pyplot(fig)

    else:
        st.warning("Silakan upload data terlebih dahulu.")

# ---- RANDOM FOREST CLASSIFICATION ----
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

# ---- PERBANDINGAN METODE ----
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
        silhouette_avg = silhouette_score(df[['income', 'score']], df['Cluster'])
        st.write(f"Silhouette Score (K-Means): {silhouette_avg:.2f}")

        if accuracy > 0.8:
            st.write("- Random Forest memiliki tingkat akurasi yang tinggi.")
        else:
            st.write("- Akurasi Random Forest masih perlu ditingkatkan.")

        if silhouette_avg > 0.5:
            st.write("- K-Means menunjukkan hasil clustering yang cukup baik.")
        else:
            st.write("- Hasil clustering dengan K-Means masih bisa ditingkatkan.")

        st.write("- Kombinasi K-Means dan Random Forest dapat digunakan untuk segmentasi pelanggan.")

    else:
        st.warning("Silakan jalankan K-Means Clustering terlebih dahulu.")

# ---- INPUT MANUAL DATA ----
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
            st.write(f"Data yang Anda masukkan termasuk dalam klaster: {cluster}")
    else:
        st.warning("Silakan upload data terlebih dahulu.")
