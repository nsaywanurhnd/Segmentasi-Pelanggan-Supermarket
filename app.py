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
tab1, tab2, tab3, tab4 = st.tabs(["📊 Visualisasi Data", "📈 K-Means", "🌲 Random Forest", "📋 Dashboard"])

# ---- Tab 1: Visualisasi Data ----
with tab1:
    st.header("📊 Visualisasi Data")
    col1, col2 = st.columns([2, 1])
    
    with col1:
        fig = px.histogram(df, x='income', title="📊 Distribusi Income Pelanggan", nbins=20)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.histogram(df, x='score', title="📊 Distribusi Spending Score", nbins=20)
        st.plotly_chart(fig, use_container_width=True)

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

    use_kmeans = st.checkbox("Gunakan hasil klaster dari K-Means sebagai target", value=True)
    
    if use_kmeans:
        y = df['Cluster']
    else:
        y = df['score']  # atau target lain yang Anda inginkan
    
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    
    st.subheader("Classification Report")
    st.text(classification_report(y_test, y_pred))
    
    cm = confusion_matrix(y_test, y_pred)
    fig = px.imshow(cm, text_auto=True, color_continuous_scale='Blues', title="Confusion Matrix")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("📊 Distribusi Kelas")
    fig = px.bar(df['Cluster'].value_counts(), title="Distribusi Kelas", labels={'value': 'Jumlah', 'index': 'Kelas'})
    st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("📊 Feature Importance")
    importances = rf.feature_importances_
    feature_names = ['income', 'score']
    fig = px.bar(x=feature_names, y=importances, title="Feature Importance", labels={'x': 'Fitur', 'y': 'Importance'})
    st.plotly_chart(fig, use_container_width=True)

# ---- Tab 4: Dashboard ----
with tab4:
    st.header("📋 Dashboard Segmentasi Pelanggan")
    cluster_filter = st.multiselect("Pilih Cluster untuk ditampilkan:", options=df['Cluster'].unique(), default=df['Cluster'].unique())
    filtered_df = df[df['Cluster'].isin(cluster_filter)]
    
    st.subheader("📋 Hasil Segmentasi Pelanggan")
    st.dataframe(filtered_df.head(20))
    
    st.subheader("📥 Unduh Laporan")
    if st.button("Unduh Hasil Klaster sebagai CSV"):
        filtered_df.to_csv('hasil_klaster.csv', index=False)
        st.success("File berhasil diunduh!")
    
    st.subheader("📊 Perbandingan Metode K-Means dan Random Forest")
    st.markdown("""
        <p>
            - <b>K-Means</b>: Metode clustering yang membagi data menjadi beberapa kelompok berdasarkan kemiripan.
            - <b>Random Forest</b>: Metode klasifikasi yang menggunakan ensemble dari banyak pohon keputusan.
        </p>
        <p>
            <b>Silhouette Score (K-Means)</b>: {:.2f}
        </p>
        <p>
            <b>Akurasi (Random Forest)</b>: {:.2f}%
        </p>
    """.format(silhouette_score(X_scaled, df['Cluster']), accuracy_score(y_test, y_pred) * 100), unsafe_allow_html=True)

# ---- Tab 5: Panduan User ----
with st.expander("ℹ️ Panduan Pengguna dan Penjelasan Hasil"):
    st.markdown("""
        ## 📘 Panduan Pengguna

        ### Langkah 1: Unggah Data
        - Pastikan file CSV memiliki kolom `income` dan `score`.
        - Jika tidak memiliki data, gunakan data sampel yang disediakan.

        ### Langkah 2: Visualisasi Data
        - Buka tab **📊 Visualisasi Data** untuk melihat distribusi pendapatan dan skor pengeluaran.

        ### Langkah 3: Segmentasi dengan K-Means
        - Buka tab **📈 K-Means**.
        - Gunakan **Elbow Method** untuk menentukan jumlah klaster.
        - Pilih jumlah klaster dan lihat hasil segmentasi.

        ### Langkah 4: Klasifikasi dengan Random Forest
        - Buka tab **🌲 Random Forest**.
        - Pilih target (hasil klaster atau kolom lain).
        - Lihat **Classification Report** dan **Confusion Matrix**.

        ### Langkah 5: Analisis Hasil di Dashboard
        - Buka tab **📋 Dashboard**.
        - Filter klaster tertentu dan unduh hasil segmentasi.

        ## 📊 Penjelasan Hasil

        ### Hasil K-Means Clustering
        - **Klaster**: Pelanggan dibagi berdasarkan kemiripan pendapatan dan skor pengeluaran.
        - **Interpretasi**:
          - **Klaster 0**: Pendapatan rendah, pengeluaran rendah.
          - **Klaster 1**: Pendapatan tinggi, pengeluaran tinggi.
          - **Klaster 2**: Pendapatan menengah, pengeluaran menengah.

        ### Hasil Random Forest
        - **Akurasi**: Menunjukkan seberapa baik model memprediksi klaster.
        - **Feature Importance**: Menunjukkan seberapa penting `income` dan `score` dalam prediksi.

        ### Cara Menggunakan Hasil
        - Gunakan hasil untuk target marketing, rekomendasi produk, dan program loyalitas.
    """)

# ---- Metrik Penting ----
st.sidebar.header("📊 Metrik Penting")
st.sidebar.metric("Total Pelanggan", df.shape[0])
st.sidebar.metric("Jumlah Klaster", df['Cluster'].nunique())
st.sidebar.metric("Akurasi Random Forest", f"{accuracy_score(y_test, y_pred) * 100:.2f}%")
