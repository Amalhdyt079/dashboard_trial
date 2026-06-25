import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. PENGATURAN HALAMAN
st.set_page_config(page_title="Dashboard Segmentasi Pelanggan", layout="wide")
st.title("🛍️ Dashboard Analitik Segmentasi Pelanggan")
st.markdown("Berdasarkan Model RFM dan Algoritma K-Means")
st.write("---")

# 2. MEMUAT DATA
@st.cache_data
def load_data():
    # Pastikan file CSV ini berada di folder yang sama dengan dashboard.py
    return pd.read_csv('rfm_hasil_clustering.csv')

try:
    df = load_data()
except FileNotFoundError:
    st.error("File 'rfm_hasil_clustering.csv' tidak ditemukan. Pastikan file tersebut ada di folder yang sama dengan dashboard ini.")
    st.stop()

# 3. METRIK UTAMA (BAGIAN ATAS)
total_cust = len(df)
total_rev = df['Monetary_Asli'].sum()
avg_freq = df['Frequency_Asli'].mean()

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Pelanggan", f"{total_cust:,}")
col2.metric("Total Pendapatan", f"Rp {total_rev:,.0f}")
col3.metric("Rata-rata Transaksi", f"{avg_freq:.1f} kali")
col4.metric("Algoritma Terbaik", "K-Means")

st.write("---")

# 4. VISUALISASI UTAMA (BAGIAN TENGAH)
col_kiri, col_kanan = st.columns(2)

with col_kiri:
    st.subheader("Distribusi Segmen Pelanggan")
    # Pie chart proporsi cluster
    cluster_counts = df['Cluster'].value_counts()
    fig_pie, ax_pie = plt.subplots(figsize=(6, 4))
    ax_pie.pie(cluster_counts, labels=[f"Cluster {i}" for i in cluster_counts.index], 
               autopct='%1.1f%%', colors=sns.color_palette('tab10'))
    st.pyplot(fig_pie)

with col_kanan:
    st.subheader("Profil Karakteristik (Snake Plot)")
    # Membentuk ulang data untuk Snake Plot
    df_melt = pd.melt(df, id_vars=['CustomerID', 'Cluster'], 
                      value_vars=['Recency_scaled', 'Frequency_scaled', 'Monetary_scaled'],
                      var_name='Metric', value_name='Value')
    
    fig_snake, ax_snake = plt.subplots(figsize=(8, 5))
    sns.lineplot(x='Metric', y='Value', hue='Cluster', data=df_melt, palette='tab10', marker='o')
    ax_snake.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    st.pyplot(fig_snake)

st.write("---")

# 5. REKOMENDASI STRATEGI BISNIS (BAGIAN BAWAH)
st.subheader("💡 Rekomendasi Strategi Pemasaran Berdasarkan Segmen")

# SESUAIKAN ISI TABEL INI DENGAN HASIL DARI BAGIAN 11 DI JUPYTER NOTEBOOK-MU!
tabel_strategi = pd.DataFrame({
    "Segmen": ["Cluster 0", "Cluster 1", "Cluster 2"], # Sesuaikan jumlahnya
    "Karakteristik Umum": [
        "TULIS KARAKTERISTIK CLUSTER 0 DI SINI (Misal: At Risk)",
        "TULIS KARAKTERISTIK CLUSTER 1 DI SINI (Misal: Loyal/Champions)",
        "TULIS KARAKTERISTIK CLUSTER 2 DI SINI (Misal: New Customers)"
    ],
    "Rekomendasi Aksi (Campaign)": [
        "Kirim diskon re-aktivasi.",
        "Berikan program loyalty/VIP.",
        "Kirimkan edukasi produk."
    ]
})

st.table(tabel_strategi)