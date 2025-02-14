import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
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

# Inisialisasi session state
if 'df' not in st.session_state:
    # Data sampel
    sample_data = pd.DataFrame({
        'income': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
        'score': [15, 25, 35, 45, 55, 65, 75, 85, 95, 5]
    })
    st.session_state.df = sample_data

# Sidebar untuk upload data
st.sidebar.header("📂 Upload Data")
uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=['csv'])

# Memuat data
if uploaded_file:
    st.session_state.df = load_data(uploaded_file)
    st.sidebar.success("Data berhasil diunggah!")
else:
    try:
        st.session_state.df = load_data("data_customer.csv")
        st.sidebar.info("Menggunakan data bawaan: data_customer.csv")
    except FileNotFoundError:
        st.sidebar.warning("Menampilkan data sampel. Silakan upload file untuk menggantinya.")

# Validasi kolom
required_columns = {'income', 'score'}
if not required_columns.issubset(st.session_state.df.columns):
    st.error(f"Dataset harus memiliki kolom: {', '.join(required_columns)}")
    st.stop()

# Sidebar untuk navigasi
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
    st.markdown("### 10 Data Pertama yang Digunakan")
    st.dataframe(st.session_state.df.head(10))
    features = [col for col in st.session_state.df.columns if col != 'id']
    selected_features = st.multiselect("Pilih fitur:", features, default=features[:2])
    if len(selected_features) >= 1:
        fig = px.histogram(st.session_state.df, x=selected_features[0], title=f"Distribusi {selected_features[0]}")
        st.plotly_chart(fig, use_container_width=True)
    if len(selected_features) >= 2:
        fig = px.histogram(st.session_state.df, x=selected_features[1], title=f"Distribusi {selected_features[1]}")
        st.plotly_chart(fig, use_container_width=True)
        fig = px.scatter(st.session_state.df, x=selected_features[0], y=selected_features[1], title=f"{selected_features[0]} vs {selected_features[1]}")
        st.plotly_chart(fig, use_container_width=True)

# ---- K-Means Clustering ----
elif menu == "📈 K-Means":
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

    X = st.session_state.df[['income', 'score']]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    inertia = calculate_inertia(X_scaled)
    fig = px.line(x=range(1, 11), y=inertia, markers=True, title="Elbow Method untuk Menentukan Jumlah Cluster")
    fig.update_layout(xaxis_title="Jumlah Cluster", yaxis_title="Inertia")
    st.plotly_chart(fig, use_container_width=True)

    num_clusters = st.slider("Pilih jumlah cluster:", 3, 11, step=2, value=3)
    if st.button("Run Clustering"):
        kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
        st.session_state.df['Cluster'] = kmeans.fit_predict(X_scaled)
        st.session_state.kmeans = kmeans  # Simpan model ke session state
        st.success("Clustering selesai! Hasil klaster telah ditambahkan ke dataset.")
        st.dataframe(st.session_state.df.head(10))
        fig = px.scatter(st.session_state.df, x='income', y='score', color=st.session_state.df['Cluster'].astype(str), title="K-Means Clustering")
        st.plotly_chart(fig, use_container_width=True)
        silhouette_avg = silhouette_score(X_scaled, st.session_state.df['Cluster'])
        st.metric("Silhouette Score", f"{silhouette_avg:.2f}")

# ---- Random Forest ----
elif menu == "🌲 Random Forest":
    st.header("🌲 Random Forest Classification")
    if 'Cluster' not in st.session_state.df.columns:
        st.error("Jalankan K-Means Clustering terlebih dahulu untuk mendapatkan kolom target.")
        st.stop()

    X = st.session_state.df[['income', 'score']]
    y = st.session_state.df['Cluster']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)

    # Simpan ke session state
    st.session_state.rf = rf
    st.session_state.y_test = y_test
    st.session_state.y_pred = y_pred

    st.markdown("### Classification Report")
    st.text(classification_report(y_test, y_pred))

    st.markdown("### Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)
    fig = px.imshow(cm, text_auto=True, color_continuous_scale='Blues', title="Confusion Matrix")
    st.plotly_chart(fig, use_container_width=True)

# ---- Dashboard ----
elif menu == "📋 Dashboard":
    st.header("📋 Dashboard Segmentasi Pelanggan")
    if 'Cluster' in st.session_state.df.columns:
        cluster_filter = st.multiselect("Pilih Cluster untuk ditampilkan:", options=st.session_state.df['Cluster'].unique(), default=st.session_state.df['Cluster'].unique())
        filtered_df = st.session_state.df[st.session_state.df['Cluster'].isin(cluster_filter)]
        st.markdown("### Hasil Segmentasi Pelanggan")
        st.dataframe(filtered_df.head(20))
        if st.button("Unduh sebagai CSV"):
            csv = filtered_df.to_csv(index=False)
            b64 = base64.b64encode(csv.encode()).decode()
            href = f'<a href="data:file/csv;base64,{b64}" download="segmentasi_pelanggan.csv">Unduh CSV</a>'
            st.markdown(href, unsafe_allow_html=True)
        silhouette_avg = silhouette_score(X_scaled, st.session_state.df['Cluster'])
        st.metric("Silhouette Score", f"{silhouette_avg:.2f}")
    else:
        st.warning("Jalankan K-Means Clustering terlebih dahulu untuk melihat hasil segmentasi.")

# ---- Perbandingan Metode ----
elif menu == "🔄 Perbandingan Metode":
    st.header("🔄 Perbandingan Metode K-Means dan Random Forest")
    if 'Cluster' in st.session_state.df.columns and 'y_test' in st.session_state and 'y_pred' in st.session_state:
        silhouette_avg = silhouette_score(X_scaled, st.session_state.df['Cluster'])
        accuracy = accuracy_score(st.session_state.y_test, st.session_state.y_pred) * 100

        st.markdown(f"- **Silhouette Score (K-Means)**: {silhouette_avg:.2f}")
        st.markdown(f"- **Akurasi (Random Forest)**: {accuracy:.2f}%")

        cm = confusion_matrix(st.session_state.y_test, st.session_state.y_pred)
        fig = px.imshow(cm, text_auto=True, color_continuous_scale='Blues', title="Confusion Matrix")
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("### Classification Report")
        st.text(classification_report(st_session_state.y_test, st.session_state.y_pred))
    else:
        st.warning("Jalankan K-Means dan Random Forest terlebih dahulu untuk melihat perbandingan.")

# ---- Input Manual ----
elif menu == "➕ Input Manual":
    st.header("➕ Input Manual Data")
    income = st.number_input("Income (Pendapatan dalam ribu dolar):", min_value=0, value=50)
    score = st.number_input("Score (Skor Pengeluaran):", min_value=0, value=50)
    if st.button("Prediksi Klaster"):
        if 'kmeans' in st.session_state:
            input_data = np.array([[income, score]])
            input_scaled = scaler.transform(input_data)
            cluster_pred = st.session_state.kmeans.predict(input_scaled)
            st.markdown(f"### Prediksi Klaster (K-Means): **{cluster_pred[0]}**")
        else:
            st.warning("Jalankan K-Means terlebih dahulu untuk melakukan prediksi.")
        if 'rf' in st.session_state:
            input_data = np.array([[income, score]])
            rf_pred = st.session_state.rf.predict(input_data)
            st.markdown(f"### Prediksi Kategori (Random Forest): **{rf_pred[0]}**")
        else:
            st.warning("Jalankan Random Forest terlebih dahulu untuk melakukan prediksi.")

# ---- Warna UI/UX ----
st.markdown(
    """
    """,
    unsafe_allow_html=True
)

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
    """,  # <-- Add the closing triple-quote here
    unsafe_allow_html=True
)
