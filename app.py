# ---- Dashboard ----
elif menu == "📋 Dashboard":
    st.header("📋 Dashboard Segmentasi Pelanggan")
    
    # Cek apakah kolom 'Cluster' sudah ada
    if 'Cluster' in df.columns:
        st.markdown("### Filter Klaster")
        cluster_filter = st.multiselect("Pilih Cluster untuk ditampilkan:", options=df['Cluster'].unique(), default=df['Cluster'].unique())
        filtered_df = df[df['Cluster'].isin(cluster_filter)]
        
        st.markdown("### Hasil Segmentasi Pelanggan")
        st.dataframe(filtered_df.head(20))
        
        st.markdown("### Unduh Laporan")
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
