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
import base64

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
    ["🏠 Beranda", "📊 Visualisasi Data", "📈 K-Means", "🌲 Random Forest", "📋 Dashboard", "🔄 Perbandingan Metode", "➕ Input Manual"]
)

# ---- Beranda ----
if menu == "🏠 Beranda":
    st.header("🏠 Selamat Datang di Aplikasi Segmentasi Pelanggan Toserba")
    st.markdown("""
        ### 📝 Tentang Aplikasi
        Aplikasi ini dirancang untuk membantu Anda melakukan segmentasi pelanggan toko serba ada (toserba) berdasarkan pendapatan (`income`) dan skor pengeluaran (`score`). 
        Dengan menggunakan metode **K-Means Clustering** dan **Random Forest Classification**, Anda dapat:
        - Membagi pelanggan ke dalam beberapa kelompok (klaster) berdasarkan kemiripan.
        - Memprediksi kategori pelanggan berdasarkan fitur yang diberikan.
        - Menganalisis hasil segmentasi dan prediksi untuk pengambilan keputusan yang lebih baik.

        ### 🛠️ Panduan Pengguna
        1. **Unggah Data**: Pastikan file CSV memiliki kolom `income` dan `score`.
        2. **Visualisasi Data**: Lihat distribusi pendapatan dan skor pengeluaran pelanggan.
        3. **K-Means Clustering**: Lakukan segmentasi pelanggan menggunakan metode K-Means.
        4. **Random Forest Classification**: Prediksi kategori pelanggan menggunakan Random Forest.
        5. **Dashboard**: Lihat hasil segmentasi dan analisis lebih lanjut.
        6. **Perbandingan Metode**: Bandingkan performa K-Means dan Random Forest.
        7. **Input Manual**: Masukkan data manual untuk prediksi klaster.

        ### 🚀 Mulai Sekarang!
        Pilih menu di sidebar untuk memulai analisis Anda.
    """)

# ---- Visualisasi Data ----
elif menu == "📊 Visualisasi Data":
    st.header("📊 Visualisasi Data")
    
    # Tampilkan 10 data pertama
    st.markdown("### 10 Data Pertama yang Digunakan")
    st.dataframe(df.head(10))
    
    # Pilih fitur untuk visualisasi
    st.markdown("### Pilih Fitur untuk Visualisasi")
    features = [col for col in df.columns if col != 'id']  # Kecuali kolom 'id'
    selected_features = st.multiselect("Pilih fitur:", features, default=features[:2])
    
    if len(selected_features) >= 1:
        st.markdown(f"### Visualisasi {selected_features[0]}")
        fig = px.histogram(df, x=selected_features[0], title=f"Distribusi {selected_features[0]}")
        st.plotly_chart(fig, use_container_width=True)
    
    if len(selected_features) >= 2:
        st.markdown(f"### Visualisasi {selected_features[1]}")
        fig = px.histogram(df, x=selected_features[1], title=f"Distribusi {selected_features[1]}")
        st.plotly_chart(fig, use_container_width=True)
    
    if len(selected_features) >= 2:
        st.markdown(f"### Scatter Plot {selected_features[0]} vs {selected_features[1]}")
        fig = px.scatter(df, x=selected_features[0], y=selected_features[1], title=f"{selected_features[0]} vs {selected_features[1]}")
        st.plotly_chart(fig, use_container_width=True)

# ---- K-Means Clustering ----
elif menu == "📈 K-Means":
    st.header("📈 K-Means Clustering")
    st.markdown("""
        ### 📝 Deskripsi Metode
        K-Means Clustering adalah metode untuk membagi data ke dalam beberapa kelompok (klaster) berdasarkan kemiripan. 
        Metode ini cocok untuk segmentasi pelanggan berdasarkan pendapatan dan skor pengeluaran.
    """)
    
    st.markdown("### Evaluasi dengan Elbow Method")
    st.markdown("""
        **Elbow Method** digunakan untuk menentukan jumlah klaster yang optimal. 
        Grafik di bawah ini menunjukkan nilai inersia (inertia) untuk berbagai jumlah klaster. 
        Pilih jumlah klaster di mana penurunan inersia mulai melambat (titik siku).
    """)
    
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
    
    st.markdown("### Pilih Jumlah Klaster")
    st.markdown("""
        Pilih jumlah klaster (nilai K) yang akan digunakan. 
        **Pastikan nilai K ganjil dan dimulai dari 3**.
    """)
    
    num_clusters = st.slider("Pilih jumlah cluster:", 3, 11, step=2, value=3)
    
    # Tombol "Run Clustering"
    if st.button("Run Clustering"):
        kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
        df['Cluster'] = kmeans.fit_predict(X_scaled)
        
        st.success("Clustering selesai! Hasil klaster telah ditambahkan ke dataset.")
        
        # Tampilkan Tabel Hasil Klaster
        st.markdown("### Tabel Hasil Klaster")
        st.dataframe(df.head(10))
        
        # Visualisasi Hasil Klaster
        st.markdown("### Visualisasi Hasil Klaster")
        fig = px.scatter(df, x='income', y='score', color=df['Cluster'].astype(str), title="K-Means Clustering", labels={'color': 'Cluster'})
        st.plotly_chart(fig, use_container_width=True)
        
        # Silhouette Score
        silhouette_avg = silhouette_score(X_scaled, df['Cluster'])
        st.markdown("### Silhouette Score")
        st.markdown(f"""
            **Silhouette Score**: {silhouette_avg:.2f}
            - Nilai berkisar antara -1 hingga 1.
            - Nilai mendekati 1 menunjukkan klaster yang baik.
        """)
        
        # Diagram Silhouette Score
        fig = px.bar(x=["Silhouette Score"], y=[silhouette_avg], title="Silhouette Score", labels={'x': 'Metrik', 'y': 'Nilai'})
        st.plotly_chart(fig, use_container_width=True)
        
        # Penjelasan Tiap Klaster
        st.markdown("### Penjelasan Tiap Klaster")
        for cluster in df['Cluster'].unique():
            cluster_data = df[df['Cluster'] == cluster]
            st.markdown(f"""
                #### Klaster {cluster}
                - **Rata-rata Income**: {cluster_data['income'].mean():.2f}
                - **Rata-rata Score**: {cluster_data['score'].mean():.2f}
                - **Jumlah Pelanggan**: {len(cluster_data)}
            """)
    else:
        st.warning("Klik tombol 'Run Clustering' untuk menjalankan proses clustering.")

# ---- Random Forest ----
elif menu == "🌲 Random Forest":
    st.header("🌲 Random Forest Classification")
    st.markdown("""
        ### 📝 Deskripsi Metode
        Random Forest adalah metode klasifikasi yang menggunakan ensemble dari banyak pohon keputusan. 
        Metode ini cocok untuk memprediksi kategori pelanggan berdasarkan fitur yang diberikan.
    """)
    
    st.markdown("### Pilih Kolom Target")
    st.markdown("""
        Pilih kolom target yang ingin diprediksi. 
        Kolom target harus berupa kategori atau nilai yang ingin diprediksi.
    """)
    
    target_column = st.selectbox("Pilih kolom target:", df.columns)
    
    X = df[['income', 'score']]
    y = df[target_column]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    
    st.markdown("### Classification Report")
    st.markdown("""
        **Classification Report** menunjukkan performa model untuk setiap kelas. 
        Metrik yang digunakan adalah precision, recall, dan F1-score.
    """)
    
    st.text(classification_report(y_test, y_pred))
    
    st.markdown("### Confusion Matrix")
    st.markdown("""
        **Confusion Matrix** menunjukkan jumlah prediksi benar dan salah untuk setiap kelas. 
        Diagonal utama menunjukkan prediksi yang benar.
    """)
    
    cm = confusion_matrix(y_test, y_pred)
    fig = px.imshow(cm, text_auto=True, color_continuous_scale='Blues', title="Confusion Matrix")
    st.plotly_chart(fig, use_container_width=True)

# ---- Dashboard ----
elif menu == "📋 Dashboard":
    st.header("📋 Dashboard Segmentasi Pelanggan")
    
    # Cek apakah kolom 'Cluster' sudah ada
    if 'Cluster' in df.columns:
        st.markdown("### Filter Klaster")
        st.markdown("""
            Pilih klaster yang ingin ditampilkan. 
            Anda dapat memilih satu atau beberapa klaster untuk dianalisis lebih lanjut.
        """)
        
        cluster_filter = st.multiselect("Pilih Cluster untuk ditampilkan:", options=df['Cluster'].unique(), default=df['Cluster'].unique())
        filtered_df = df[df['Cluster'].isin(cluster_filter)]
        
        st.markdown("### Hasil Segmentasi Pelanggan")
        st.dataframe(filtered_df.head(20))
        
        st.markdown("### Unduh Laporan")
        st.markdown("""
            Klik tombol di bawah ini untuk mengunduh hasil segmentasi dalam format CSV.
        """)
        
        if st.button("Unduh sebagai CSV"):
            csv = filtered_df.to_csv(index=False)
            b64 = base64.b64encode(csv.encode()).decode()
            href = f'<a href="data:file/csv;base64,{b64}" download="hasil_klaster.csv">Unduh CSV</a>'
            st.markdown(href, unsafe_allow_html=True)
        
        # Visualisasi Silhouette Score
        st.markdown("### Silhouette Score")
        st.markdown("""
            **Silhouette Score** mengukur seberapa baik data terpisah ke dalam klaster. 
            Nilai berkisar antara -1 hingga 1, di mana nilai mendekati 1 menunjukkan klaster yang baik.
        """)
        
        silhouette_avg = silhouette_score(X_scaled, df['Cluster'])
        st.metric("Silhouette Score", f"{silhouette_avg:.2f}")
        
        # Diagram Silhouette Score
        fig = px.bar(x=["Silhouette Score"], y=[silhouette_avg], title="Silhouette Score", labels={'x': 'Metrik', 'y': 'Nilai'})
        st.plotly_chart(fig, use_container_width=True)
        
        # Penjelasan Hasil Klaster
        st.markdown("### Penjelasan Hasil Klaster")
        for cluster in df['Cluster'].unique():
            cluster_data = df[df['Cluster'] == cluster]
            st.markdown(f"""
                #### Klaster {cluster}
                - **Rata-rata Income**: {cluster_data['income'].mean():.2f}
                - **Rata-rata Score**: {cluster_data['score'].mean():.2f}
                - **Jumlah Pelanggan**: {len(cluster_data)}
            """)
    else:
        st.warning("Jalankan K-Means Clustering terlebih dahulu untuk melihat hasil segmentasi.")

# ---- Perbandingan Metode ----
elif menu == "🔄 Perbandingan Metode":
    st.header("🔄 Perbandingan Metode K-Means dan Random Forest")
    
    # Cek apakah K-Means dan Random Forest sudah dijalankan
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
        
        # Visualisasi Confusion Matrix
        st.markdown("### Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred)
        fig = px.imshow(cm, text_auto=True, color_continuous_scale='Blues', title="Confusion Matrix")
        st.plotly_chart(fig, use_container_width=True)
        
        # Classification Report
        st.markdown("### Classification Report")
        st.text(classification_report(y_test, y_pred))
    else:
        st.warning("Jalankan K-Means dan Random Forest terlebih dahulu untuk melihat perbandingan.")

# ---- Input Manual ----
elif menu == "➕ Input Manual":
    st.header("➕ Input Manual Data")
    st.markdown("""
        ### 📝 Masukkan Data Manual
        Masukkan nilai `income` dan `score` untuk memprediksi klaster menggunakan K-Means dan Random Forest.
    """)
    
    income = st.number_input("Income (Pendapatan dalam ribu dolar):", min_value=0, value=50)
    score = st.number_input("Score (Skor Pengeluaran):", min_value=0, value=50)
    
    if st.button("Prediksi Klaster"):
        # Prediksi menggunakan K-Means
        if 'kmeans' in locals():
            input_data = np.array([[income, score]])
            input_scaled = scaler.transform(input_data)
            cluster_pred = kmeans.predict(input_scaled)
            st.markdown(f"### Prediksi Klaster (K-Means): **{cluster_pred[0]}**")
        
        # Prediksi menggunakan Random Forest
        if 'rf' in locals():
            input_data = np.array([[income, score]])
            rf_pred = rf.predict(input_data)
            st.markdown(f"### Prediksi Kategori (Random Forest): **{rf_pred[0]}**")
        else:
            st.warning("Jalankan Random Forest terlebih dahulu untuk melakukan prediksi.")

# ---- Warna UI/UX ----
st.markdown(
    """
    <style>
    .stTabs [role="tablist"] { 
        justify-content: center;
        margin-top: 50px;
    }
    .stTabs [role="tab"] { 
        font-size: 20px; 
        font-weight: bold; 
        padding: 15px 25px; 
        border-radius: 8px;
        background-color: #f0f2f6;
        color: #333;
    }
    .stTabs [role="tab"]:hover { 
        color: white; 
        background-color: #007bff; 
    }
    .stTabs [role="tab"][aria-selected="true"] { 
        color: white; 
        background-color: #007bff; 
    }
    .stButton button {
        background-color: #007bff;
        color: white;
        font-weight: bold;
        border-radius: 8px;
        padding: 10px 20px;
    }
    .stButton button:hover {
        background-color: #0056b3;
    }
    </style>
    """,
    unsafe_allow_html=True
)
