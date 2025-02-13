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

# Sidebar untuk navigasi
st.sidebar.header("📑 Menu Navigasi")
menu = st.sidebar.radio(
    "Pilih Menu:",
    ["📊 Visualisasi Data", "📈 K-Means", "🌲 Random Forest", "📋 Dashboard"]
)

# ---- Visualisasi Data ----
if menu == "📊 Visualisasi Data":
    st.header("📊 Visualisasi Data")
    col1, col2 = st.columns([2, 1])
    
    with col1:
        fig = px.line(df, x=df.index, y='income', title="📈 Tren Income Pelanggan")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.pie(df, names="score", title="🎯 Distribusi Spending Score", hole=0.4)
        st.plotly_chart(fig, use_container_width=True)

# ---- Tab 2: K-Means Clustering ----
if menu == "📈 K-Means":
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
    
    X = df[['income', 'score']]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    inertia = calculate_inertia(X_scaled)
    fig = px.line(x=range(1, 11), y=inertia, markers=True, title="Elbow Method untuk Menentukan Jumlah Cluster")
    fig.update_layout(xaxis_title="Jumlah Cluster", yaxis_title="Inertia")
    st.plotly_chart(fig, use_container_width=True)
    
    # Validasi nilai K (harus ganjil dan dimulai dari 3)
    num_clusters = st.slider("Pilih jumlah cluster:", 3, 11, step=2, value=3)
    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
    df['Cluster'] = kmeans.fit_predict(X_scaled)
    
    # Buat list figures untuk menyimpan grafik
    figures = []
    
    # Grafik 1: Scatter plot untuk K-Means
    fig1 = px.scatter(df, x='income', y='score', color=df['Cluster'].astype(str), title="K-Means Clustering")
    figures.append(fig1)
    
    # Grafik 2: Histogram untuk distribusi income
    fig2 = px.histogram(df, x='income', title="Distribusi Income")
    figures.append(fig2)
    
    # Tampilkan semua grafik
    for i, fig in enumerate(figures):
        st.plotly_chart(fig, use_container_width=True)

# ---- Random Forest ----
elif menu == "🌲 Random Forest":
    st.header("🌲 Random Forest Classification")
    st.markdown("""
        <h3 style='color: #007bff; font-size: 24px;'>
            📝 Deskripsi Model
        </h3>
        <p>
            Random Forest adalah metode klasifikasi yang menggunakan ensemble dari banyak pohon keputusan. 
            Model ini bekerja dengan membangun banyak pohon keputusan dan menggabungkan hasilnya untuk meningkatkan akurasi dan mengurangi overfitting.
        </p>
        <h3 style='color: #007bff; font-size: 24px;'>
            📊 Metrik yang Digunakan
        </h3>
        <p>
            - <b>Akurasi</b>: Proporsi prediksi yang benar dari total prediksi.
            - <b>Confusion Matrix</b>: Menunjukkan jumlah prediksi benar dan salah untuk setiap kelas.
            - <b>Classification Report</b>: Menampilkan precision, recall, dan F1-score untuk setiap kelas.
        </p>
        <h3 style='color: #007bff; font-size: 24px;'>
            📈 Cara Interpretasi Hasil
        </h3>
        <p>
            - <b>Akurasi</b>: Semakin tinggi akurasi, semakin baik model dalam memprediksi kelas.
            - <b>Confusion Matrix</b>: Diagonal utama menunjukkan prediksi yang benar.
            - <b>Classification Report</b>: Precision tinggi berarti sedikit false positives, recall tinggi berarti sedikit false negatives.
        </p>
    """, unsafe_allow_html=True)

    # Pilih kolom target
    target_column = st.selectbox("Pilih kolom target:", df.columns)
    
    X = df[['income', 'score']]
    y = df[target_column]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    
    st.subheader("Classification Report")
    st.text(classification_report(y_test, y_pred))
    
    cm = confusion_matrix(y_test, y_pred)
    fig = px.imshow(cm, text_auto=True, color_continuous_scale='Blues', title="Confusion Matrix")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("📊 Feature Importance")
    importances = rf.feature_importances_
    feature_names = ['income', 'score']
    fig = px.bar(x=feature_names, y=importances, title="Feature Importance", labels={'x': 'Fitur', 'y': 'Importance'})
    st.plotly_chart(fig, use_container_width=True)

# ---- Dashboard ----
elif menu == "📋 Dashboard":
    st.header("📋 Dashboard Segmentasi Pelanggan")
    
    if 'Cluster' in df.columns:
        cluster_filter = st.multiselect("Pilih Cluster untuk ditampilkan:", options=df['Cluster'].unique(), default=df['Cluster'].unique())
        filtered_df = df[df['Cluster'].isin(cluster_filter)]
        
        st.subheader("📋 Hasil Segmentasi Pelanggan")
        st.dataframe(filtered_df.head(20))
        
        st.subheader("📥 Unduh Laporan")
        if st.button("Unduh Hasil Klaster sebagai CSV"):
            filtered_df.to_csv('hasil_klaster.csv', index=False)
            st.success("File berhasil diunduh!")
    else:
        st.warning("Jalankan K-Means Clustering terlebih dahulu untuk melihat hasil segmentasi.")

# ---- Tab 5: Panduan User ----
with st.expander("ℹ️ Panduan Pengguna"):
    st.markdown("""
        ## 📘 Panduan Pengguna

        ### Langkah 1: Unggah Data
        - Pastikan file CSV memiliki kolom `income` dan `score`.
        - Jika tidak memiliki data, gunakan data sampel yang disediakan.

        ### Langkah 2: Visualisasi Data
        - Buka menu **📊 Visualisasi Data** untuk melihat distribusi pendapatan dan skor pengeluaran.

        ### Langkah 3: Segmentasi dengan K-Means
        - Buka menu **📈 K-Means**.
        - Gunakan **Elbow Method** untuk menentukan jumlah klaster.
        - Pilih jumlah klaster (nilai K harus ganjil dan dimulai dari 3).
        - Lihat hasil segmentasi pada scatter plot.

        ### Langkah 4: Klasifikasi dengan Random Forest
        - Buka menu **🌲 Random Forest**.
        - Pilih kolom target yang ingin diprediksi.
        - Lihat **Classification Report** dan **Confusion Matrix**.

        ### Langkah 5: Analisis Hasil di Dashboard
        - Buka menu **📋 Dashboard**.
        - Filter klaster tertentu dan unduh hasil segmentasi.

        ### Langkah 6: Perbandingan Metode
        - Buka menu **🔄 Perbandingan Metode** untuk melihat perbandingan antara K-Means dan Random Forest.
    """)

# ---- Tab 6: Perbandingan Metode ----
if menu == "🔄 Perbandingan Metode":
    st.header("🔄 Perbandingan Metode K-Means dan Random Forest")
    
    if 'Cluster' in df.columns and 'y_test' in locals() and 'y_pred' in locals():
        st.markdown("""
            ### 📊 Hasil K-Means Clustering
            - **Silhouette Score**: {:.2f}
            - **Inertia**: {:.2f}

            ### 📊 Hasil Random Forest
            - **Akurasi**: {:.2f}%
            - **Precision, Recall, F1-Score**: Lihat di menu **🌲 Random Forest**.

            ### 📝 Kesimpulan
            - **K-Means** cocok untuk segmentasi data berdasarkan kemiripan.
            - **Random Forest** cocok untuk prediksi kelas atau nilai target.
        """.format(
            silhouette_score(X_scaled, df['Cluster']),
            kmeans.inertia_ if 'kmeans' in locals() else "Belum dihitung",
            accuracy_score(y_test, y_pred) * 100
        ))
    else:
        st.warning("Jalankan K-Means dan Random Forest terlebih dahulu untuk melihat perbandingan.")

# ---- Validasi Nilai K di K-Means ----
if menu == "📈 K-Means":
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
    
    X = df[['income', 'score']]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    inertia = calculate_inertia(X_scaled)
    fig = px.line(x=range(1, 11), y=inertia, markers=True, title="Elbow Method untuk Menentukan Jumlah Cluster")
    fig.update_layout(xaxis_title="Jumlah Cluster", yaxis_title="Inertia")
    st.plotly_chart(fig, use_container_width=True)
    
    # Validasi nilai K (harus ganjil dan dimulai dari 3)
    num_clusters = st.slider("Pilih jumlah cluster:", 3, 11, step=2, value=3)
    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
    df['Cluster'] = kmeans.fit_predict(X_scaled)
    
    fig = px.scatter(df, x='income', y='score', color=df['Cluster'].astype(str), title="K-Means Clustering", labels={'color': 'Cluster'})
    st.plotly_chart(fig, use_container_width=True)


# ---- Metrik Penting ----
st.sidebar.header("📊 Metrik Penting")
st.sidebar.metric("Total Pelanggan", df.shape[0])

if 'Cluster' in df.columns:
    st.sidebar.metric("Jumlah Klaster", df['Cluster'].nunique())
else:
    st.sidebar.metric("Jumlah Klaster", "Belum dihitung")

if 'y_test' in locals() and 'y_pred' in locals():
    st.sidebar.metric("Akurasi Random Forest", f"{accuracy_score(y_test, y_pred) * 100:.2f}%")
else:
    st.sidebar.metric("Akurasi Random Forest", "Belum dihitung")
