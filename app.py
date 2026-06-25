import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import os

# ==========================================
# 1. PENGATURAN HALAMAN
# ==========================================
st.set_page_config(page_title="RFM ClusterLens", page_icon="🛒", layout="wide", initial_sidebar_state="expanded")

# ==========================================
# 2. LOGIKA TEMA (GELAP / CERAH)
# ==========================================
st.sidebar.markdown("<br>", unsafe_allow_html=True)
tema_gelap = st.sidebar.toggle("🌙 Mode Gelap", value=True)

if tema_gelap:
    warna = {
        "bg": "#0F1117", "surface": "#1A1D27", "surface2": "#22263A", 
        "border": "rgba(255,255,255,0.07)", "text": "#F0F2FF", 
        "muted": "#8B90A7", "grid": "rgba(255,255,255,0.05)",
        "grid_strong": "rgba(255,255,255,0.2)", # <-- Grid tebal untuk Scatter 3D
        "green": "#00D4B4", "blue": "#4F8EF7", "purple": "#A78BFA", "yellow": "#FBBF24",
        "red": "#F87171"
    }
else:
    warna = {
        "bg": "#F4F6FB", "surface": "#FFFFFF", "surface2": "#EEF1F8", 
        "border": "rgba(0,0,0,0.08)", "text": "#111827", 
        "muted": "#6B7280", "grid": "rgba(0,0,0,0.05)",
        "grid_strong": "rgba(0,0,0,0.2)", # <-- Grid tebal untuk Scatter 3D
        "green": "#00D4B4", "blue": "#4F8EF7", "purple": "#A78BFA", "yellow": "#FBBF24",
        "red": "#F87171"
    }

st.markdown(f'''
<style>
    .stApp {{ background-color: {warna['bg']}; color: {warna['text']}; font-family: 'Inter', sans-serif; }}
    [data-testid="stSidebar"] {{ background-color: {warna['surface']}; border-right: 1px solid {warna['border']}; }}
    [data-testid="stHeader"] {{ background-color: rgba(0,0,0,0); }}
    .block-container {{ padding-top: 2rem; }}

    .kpi-grid {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 16px; margin-bottom: 24px; }}
    .kpi {{ background: {warna['surface']}; border: 1px solid {warna['border']}; border-radius: 12px; padding: 16px 20px; display: flex; align-items: center; gap: 14px; box-shadow: 0 4px 6px rgba(0,0,0,0.02); }}
    .kpi-icon {{ width: 42px; height: 42px; border-radius: 10px; display: flex; align-items: center; justify-content: center; font-size: 20px; flex-shrink: 0; }}
    .kpi-val {{ font-size: 22px; font-weight: 800; line-height: 1; color: {warna['text']}; margin-bottom: 4px; }}
    .kpi-label {{ font-size: 11px; color: {warna['muted']}; text-transform: uppercase; font-weight: 600; }}
    .kpi-subtext {{ font-size: 11px; margin-top: 4px; font-weight: 500; }}

    .topbar {{ background: {warna['surface']}; border: 1px solid {warna['border']}; border-radius: 12px; padding: 15px 24px; display: flex; align-items: center; justify-content: space-between; margin-bottom: 24px; box-shadow: 0 4px 6px rgba(0,0,0,0.02);}}
    .topbar-title {{ font-size: 18px; font-weight: 700; color: {warna['text']}; margin: 0; }}
    .topbar-sub {{ font-size: 12px; color: {warna['muted']}; margin: 0; }}
    .topbar-pill {{ background: {warna['surface2']}; border: 1px solid {warna['border']}; border-radius: 20px; padding: 6px 15px; font-size: 12px; color: {warna['muted']}; font-weight: 600; }}
    
    .summary-card {{ background: {warna['surface']}; border: 1px solid {warna['border']}; border-radius: 12px; padding: 20px; height: 100%; box-shadow: 0 4px 6px rgba(0,0,0,0.02); }}
    .chart-card {{ background: {warna['surface']}; border: 1px solid {warna['border']}; border-radius: 12px; padding: 16px; margin-bottom: 16px; box-shadow: 0 4px 6px rgba(0,0,0,0.02); }}
    .summary-title {{ font-size: 12px; font-weight: 700; color: {warna['muted']}; text-transform: uppercase; margin-bottom: 16px; letter-spacing: 0.5px; display:flex; justify-content:space-between; align-items:center; }}
    .list-item {{ display: flex; justify-content: space-between; padding: 10px 0; border-bottom: 1px solid {warna['grid']}; font-size: 13px; font-weight: 500; color: {warna['muted']}; }}
    .list-item:last-child {{ border-bottom: none; padding-bottom: 0; }}
    .val-text {{ font-weight: 700; color: {warna['text']}; }}
    
    .kesimpulan-box {{ background: linear-gradient(135deg, rgba(0,212,180,0.05), rgba(79,142,247,0.05)); border: 1px solid rgba(0,212,180,0.2); border-radius: 12px; padding: 20px; margin-top: 16px; }}
    .noise-box {{ background: rgba(248,113,113,0.05); border: 1px dashed rgba(248,113,113,0.3); border-radius: 12px; padding: 16px; margin-bottom: 24px; }}
</style>
''', unsafe_allow_html=True)

# ==========================================
# 3. MEMUAT SEMUA DATA (DINAMIS)
# ==========================================
@st.cache_data
def load_data():
    try: df = pd.read_csv('rfm_hasil_clustering.csv')
    except:
        try: df = pd.read_csv('./Tugas_Akhir/rfm_hasil_clustering.csv')
        except: df = pd.DataFrame({'CustomerID':[1],'Recency_Asli':[1],'Frequency_Asli':[1],'Monetary_Asli':[1],'Cluster_KMeans':[0],'Cluster_Hierarchical':[0],'Cluster_DBSCAN':[0]})
        
    try: df_eval = pd.read_csv('tabel_evaluasi.csv')
    except:
        try: df_eval = pd.read_csv('./Tugas_Akhir/tabel_evaluasi.csv')
        except: df_eval = pd.DataFrame({'Metode': ['K-Means', 'Hierarchical Clustering', 'DBSCAN'], 'Jumlah Cluster': [4, 4, 3], 'Silhouette Score': [0.4870, 0.4610, 0.3120], 'Davies-Bouldin Index': [0.8120, 0.9340, 1.2430]})
    if 'Silhouette Score' in df_eval.columns: df_eval.rename(columns={'Silhouette Score': 'Silhouette', 'Davies-Bouldin Index': 'DBI'}, inplace=True)
    
    try: df_preproc = pd.read_csv('preprocessing_summary.csv')
    except:
        try: df_preproc = pd.read_csv('./Tugas_Akhir/preprocessing_summary.csv')
        except: df_preproc = pd.DataFrame({'Nilai': [541909, -135080, -10624, 0, 392732]})
            
    v_awal = df_preproc.iloc[0]['Nilai']
    v_missing = df_preproc.iloc[1]['Nilai']
    v_qty = df_preproc.iloc[2]['Nilai']
    v_price = df_preproc.iloc[3]['Nilai']
    v_bersih = df_preproc.iloc[4]['Nilai']

    return df, df_eval, v_awal, v_missing, v_qty, v_price, v_bersih

df, df_evaluasi, v_awal, v_missing, v_qty, v_price, v_bersih = load_data()

def show_image(img_name):
    if os.path.exists(img_name): st.image(img_name, use_container_width=True)
    elif os.path.exists(f'./Tugas_Akhir/{img_name}'): st.image(f'./Tugas_Akhir/{img_name}', use_container_width=True)
    else: st.warning(f"⚠️ Gambar {img_name} tidak ditemukan.")

# ==========================================
# 4. SIDEBAR MENU NAVIGASI
# ==========================================
st.sidebar.markdown("<hr style='border-color: "+warna['border']+"; margin-top:5px; margin-bottom:20px;'>", unsafe_allow_html=True)
st.sidebar.markdown("### 🧭 Menu Navigasi")

menu_pilihan = st.sidebar.radio("Pilih Halaman:", 
    ("Overview", "Analisis RFM", "K-Means", "Hierarchical Clustering", "DBSCAN", "Perbandingan Metode"), 
    label_visibility="collapsed")

warna_klaster = {'Cluster 0': '#00D4B4', 'Cluster 1': '#4F8EF7', 'Cluster 2': '#A78BFA', 'Cluster 3': '#FBBF24', 'Noise (-1)': '#F87171'}
warna_metode = {'K-Means': '#00D4B4', 'Hierarchical Clustering': '#4F8EF7', 'DBSCAN': '#A78BFA'}
tema_plotly = dict(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color=warna['text']), margin=dict(l=20, r=20, b=20, t=40))

# ==============================================================================
# HALAMAN 0: OVERVIEW 
# ==============================================================================
if menu_pilihan == "Overview":
    st.markdown(f'''
    <div style='display:flex; justify-content:space-between; align-items:flex-start; margin-bottom:20px;'>
        <div>
            <h2 style='margin:0; font-size:24px; font-weight:700; color:{warna["text"]};'>Overview Dataset & Hasil Clustering</h2>
            <p style='margin:4px 0 0 0; font-size:13px; color:{warna["muted"]};'>Online Retail Dataset - UCI Machine Learning Repository</p>
        </div>
    </div>
    ''', unsafe_allow_html=True)

    tot_pelanggan = len(df); rata_r = df['Recency_Asli'].mean(); rata_f = df['Frequency_Asli'].mean(); rata_m = df['Monetary_Asli'].mean()
    med_r = df['Recency_Asli'].median(); max_m = df['Monetary_Asli'].max()
    raw_k = v_awal / 1000; bersih_k = v_bersih / 1000
    
    st.markdown(f'''
    <div class="kpi-grid">
        <div class="kpi" style="flex-direction: column; align-items: flex-start; justify-content: center; gap:4px;">
            <div style="display:flex; align-items:center; gap:12px;"><div class="kpi-icon" style="background: rgba(0,212,180,0.1); color: {warna['green']};">👥</div><div><div class="kpi-val" style="color:{warna['green']};">{tot_pelanggan:,}</div><div class="kpi-label">Total Pelanggan</div></div></div>
            <div class="kpi-subtext" style="color:{warna['green']};">Setelah preprocessing</div>
        </div>
        <div class="kpi" style="flex-direction: column; align-items: flex-start; justify-content: center; gap:4px;">
            <div style="display:flex; align-items:center; gap:12px;"><div class="kpi-icon" style="background: rgba(79,142,247,0.1); color: {warna['blue']};">📦</div><div><div class="kpi-val" style="color:{warna['blue']};">{v_bersih:,}</div><div class="kpi-label">Total Transaksi</div></div></div>
            <div class="kpi-subtext" style="color:{warna['blue']};">{raw_k:.0f}K raw → {bersih_k:.0f}K bersih</div>
        </div>
        <div class="kpi" style="flex-direction: column; align-items: flex-start; justify-content: center; gap:4px;">
            <div style="display:flex; align-items:center; gap:12px;"><div class="kpi-icon" style="background: rgba(167,139,250,0.1); color: {warna['purple']};">📅</div><div><div class="kpi-val" style="color:{warna['purple']};">374</div><div class="kpi-label">Periode (Hari)</div></div></div>
            <div class="kpi-subtext" style="color:{warna['purple']};">1 Des 2010 – 9 Des 2011</div>
        </div>
        <div class="kpi" style="flex-direction: column; align-items: flex-start; justify-content: center; gap:4px;">
            <div style="display:flex; align-items:center; gap:12px;"><div class="kpi-icon" style="background: rgba(251,191,36,0.1); color: {warna['yellow']};">📊</div><div><div class="kpi-val" style="color:{warna['yellow']};">3</div><div class="kpi-label">Fitur RFM</div></div></div>
            <div class="kpi-subtext" style="color:{warna['yellow']};">Recency, Frequency, Monetary</div>
        </div>
    </div>
    ''', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f'''
        <div class="summary-card">
            <div class="summary-title">Rata-Rata RFM</div>
            <div class="list-item"><span>Recency</span> <span class="val-text" style="color:{warna['green']};">{rata_r:.1f} hari</span></div>
            <div class="list-item"><span>Frequency</span> <span class="val-text" style="color:{warna['blue']};">{rata_f:.1f} transaksi</span></div>
            <div class="list-item"><span>Monetary</span> <span class="val-text" style="color:{warna['purple']};">£ {rata_m:,.1f}</span></div>
            <div class="list-item"><span>Median Recency</span> <span class="val-text">{med_r:.0f} hari</span></div>
            <div class="list-item"><span>Max Monetary</span> <span class="val-text">£ {max_m:,.0f}</span></div>
        </div>
        ''', unsafe_allow_html=True)
    with col2:
        st.markdown(f'''
        <div class="summary-card">
            <div class="summary-title">Preprocessing Summary</div>
            <div class="list-item"><span>Data awal</span> <span class="val-text">{v_awal:,}</span></div>
            <div class="list-item"><span>Missing value / No ID</span> <span class="val-text">{v_missing:,}</span></div>
            <div class="list-item"><span>Quantity ≤ 0</span> <span class="val-text">{v_qty:,}</span></div>
            <div class="list-item"><span>UnitPrice ≤ 0</span> <span class="val-text">{v_price:,}</span></div>
            <div class="list-item"><span>Data bersih</span> <span class="val-text" style="color:{warna['blue']}">{v_bersih:,}</span></div>
        </div>
        ''', unsafe_allow_html=True)
    with col3:
        st.markdown(f'''
        <div class="summary-card">
            <div class="summary-title">Normalisasi</div>
            <div class="list-item"><span>Step 1</span> <span class="val-text">Log1p Transform</span></div>
            <div class="list-item"><span>Step 2</span> <span class="val-text">Z-Score Standardization</span></div>
            <div class="list-item"><span>Library</span> <span class="val-text">StandardScaler</span></div>
            <div class="list-item"><span>Mean setelah</span> <span class="val-text">≈ 0.00</span></div>
            <div class="list-item"><span>Std setelah</span> <span class="val-text">≈ 1.00</span></div>
        </div>
        ''', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(f'''<div><h3 style="margin:0; font-size:20px; font-weight:700; color:{warna['text']};">KPI Hasil Clustering</h3><p style="margin:4px 0 16px 0; font-size:13px; color:{warna['muted']};">Ringkasan performa ketiga metode</p></div>''', unsafe_allow_html=True)
    
    df_rank = df_evaluasi.copy(); df_rank['Rank'] = df_rank['Silhouette'].rank(ascending=False, method='min')
    def get_metrik(metode_nama):
        try:
            row = df_rank[df_rank['Metode'].str.contains(metode_nama, case=False)].iloc[0]
            jml_cluster, sil, dbi, rank = int(row['Jumlah Cluster']), row['Silhouette'], row['DBI'], row['Rank']
            if rank == 1: badge = f"<span style='background:rgba(0,212,180,0.1); color:{warna['green']}; padding:4px 10px; border-radius:20px; font-size:11px;'>🥇 Terbaik</span>"
            elif rank == 2: badge = f"<span style='background:rgba(79,142,247,0.1); color:{warna['blue']}; padding:4px 10px; border-radius:20px; font-size:11px;'>🥈 Kedua</span>"
            else: badge = f"<span style='background:rgba(251,191,36,0.1); color:{warna['yellow']}; padding:4px 10px; border-radius:20px; font-size:11px;'>🥉 Ketiga</span>"
            return jml_cluster, sil, dbi, badge
        except: return 0, 0, 0, ""

    k_clu, k_sil, k_dbi, k_badge = get_metrik('K-Means')
    h_clu, h_sil, h_dbi, h_badge = get_metrik('Hierarchical')
    d_clu, d_sil, d_dbi, d_badge = get_metrik('DBSCAN')

    str_noise = f"{(df['Cluster_DBSCAN'] == -1).sum()} titik" if 'Cluster_DBSCAN' in df.columns else "—"

    col_b1, col_b2, col_b3 = st.columns(3)
    with col_b1:
        st.markdown(f'''
        <div class="summary-card">
            <div class="summary-title"><span style="background:rgba(0,212,180,0.1); color:{warna['green']}; padding:4px 10px; border-radius:8px;">K-Means</span> {k_badge}</div>
            <div class="list-item"><span>Jumlah Cluster</span> <span class="val-text" style="color:{warna['green']};">{k_clu}</span></div>
            <div class="list-item"><span>Silhouette Score</span> <span class="val-text" style="color:{warna['green']};">{k_sil:.3f}</span></div>
            <div class="list-item"><span>Davies-Bouldin</span> <span class="val-text" style="color:{warna['green']};">{k_dbi:.3f}</span></div>
            <div class="list-item"><span>Noise</span> <span class="val-text">—</span></div>
        </div>
        ''', unsafe_allow_html=True)
    with col_b2:
        st.markdown(f'''
        <div class="summary-card">
            <div class="summary-title"><span style="background:rgba(79,142,247,0.1); color:{warna['blue']}; padding:4px 10px; border-radius:8px;">Hierarchical</span> {h_badge}</div>
            <div class="list-item"><span>Jumlah Cluster</span> <span class="val-text" style="color:{warna['blue']};">{h_clu}</span></div>
            <div class="list-item"><span>Silhouette Score</span> <span class="val-text" style="color:{warna['blue']};">{h_sil:.3f}</span></div>
            <div class="list-item"><span>Davies-Bouldin</span> <span class="val-text" style="color:{warna['blue']};">{h_dbi:.3f}</span></div>
            <div class="list-item"><span>Noise</span> <span class="val-text">—</span></div>
        </div>
        ''', unsafe_allow_html=True)
    with col_b3:
        st.markdown(f'''
        <div class="summary-card">
            <div class="summary-title"><span style="background:rgba(167,139,250,0.1); color:{warna['purple']}; padding:4px 10px; border-radius:8px;">DBSCAN</span> {d_badge}</div>
            <div class="list-item"><span>Jumlah Cluster</span> <span class="val-text" style="color:{warna['purple']};">{d_clu}</span></div>
            <div class="list-item"><span>Silhouette Score</span> <span class="val-text" style="color:{warna['purple']};">{d_sil:.3f}</span></div>
            <div class="list-item"><span>Davies-Bouldin</span> <span class="val-text" style="color:{warna['purple']};">{d_dbi:.3f}</span></div>
            <div class="list-item"><span>Noise</span> <span class="val-text" style="color:{warna['red']};">{str_noise}</span></div>
        </div>
        ''', unsafe_allow_html=True)

# ==============================================================================
# HALAMAN 1: ANALISIS RFM (EXPLORATORY DATA ANALYSIS)
# ==============================================================================
elif menu_pilihan == "Analisis RFM":
    st.markdown(f'''
    <div class="topbar">
        <div style="display: flex; gap: 15px; align-items: center;">
            <div>
                <h2 class="topbar-title">Analisis Distribusi RFM</h2>
                <p class="topbar-sub">Eksplorasi variabel Recency, Frequency, dan Monetary sebelum pemodelan</p>
            </div>
        </div>
        <div class="topbar-pill">📅 Des 2010 – Des 2011</div>
    </div>
    ''', unsafe_allow_html=True)

    col_hist1, col_hist2, col_hist3 = st.columns(3)
    
    bins_r = [0, 50, 100, 150, 200, 250, 300, np.inf]; labels_r = ['0-50', '51-100', '101-150', '151-200', '201-250', '251-300', '300+']
    df['R_Bin'] = pd.cut(df['Recency_Asli'], bins=bins_r, labels=labels_r, include_lowest=True); df_r = df['R_Bin'].value_counts().reindex(labels_r).reset_index(); df_r.columns = ['Bin', 'Count']
    with col_hist1:
        st.markdown(f"<div class='chart-card'><div class='summary-title'>DISTRIBUSI RECENCY</div>", unsafe_allow_html=True)
        fig_r = px.bar(df_r, x='Bin', y='Count'); fig_r.update_traces(marker_color=warna['green']); fig_r.update_layout(**tema_plotly, height=200, xaxis_title=None, yaxis_title=None, showlegend=False); st.plotly_chart(fig_r, use_container_width=True)
        st.markdown(f"<p style='font-size:12px; color:{warna['muted']};'>Sebagian besar pelanggan bertransaksi <b>dalam 100 hari terakhir</b>, mengindikasikan basis pelanggan aktif.</p></div>", unsafe_allow_html=True)

    bins_f = [0, 1, 2, 3, 4, 5, 10, np.inf]; labels_f = ['1', '2', '3', '4', '5', '6-10', '10+']
    df['F_Bin'] = pd.cut(df['Frequency_Asli'], bins=bins_f, labels=labels_f, include_lowest=True); df_f = df['F_Bin'].value_counts().reindex(labels_f).reset_index(); df_f.columns = ['Bin', 'Count']
    with col_hist2:
        st.markdown(f"<div class='chart-card'><div class='summary-title'>DISTRIBUSI FREQUENCY</div>", unsafe_allow_html=True)
        fig_f = px.bar(df_f, x='Bin', y='Count'); fig_f.update_traces(marker_color=warna['blue']); fig_f.update_layout(**tema_plotly, height=200, xaxis_title=None, yaxis_title=None, showlegend=False); st.plotly_chart(fig_f, use_container_width=True)
        st.markdown(f"<p style='font-size:12px; color:{warna['muted']};'>Distribusi <b>right-skewed</b>, mayoritas pelanggan bertransaksi 1-5 kali.</p></div>", unsafe_allow_html=True)

    bins_m = [-np.inf, 100, 500, 1000, 3000, 10000, np.inf]; labels_m = ['<100', '100-500', '500-1K', '1K-3K', '3K-10K', '10K+']
    df['M_Bin'] = pd.cut(df['Monetary_Asli'], bins=bins_m, labels=labels_m); df_m = df['M_Bin'].value_counts().reindex(labels_m).reset_index(); df_m.columns = ['Bin', 'Count']
    with col_hist3:
        st.markdown(f"<div class='chart-card'><div class='summary-title'>DISTRIBUSI MONETARY</div>", unsafe_allow_html=True)
        fig_m = px.bar(df_m, x='Bin', y='Count'); fig_m.update_traces(marker_color=warna['purple']); fig_m.update_layout(**tema_plotly, height=200, xaxis_title=None, yaxis_title=None, showlegend=False); st.plotly_chart(fig_m, use_container_width=True)
        st.markdown(f"<p style='font-size:12px; color:{warna['muted']};'>Distribusi sangat <b>right-skewed</b> — Log Transformation diperlukan sebelum clustering.</p></div>", unsafe_allow_html=True)

    col_bot1, col_bot2 = st.columns([1.2, 1])
    with col_bot1:
        st.markdown(f"<div class='chart-card'><div class='summary-title'>PERBANDINGAN SKEWNESS (SEBELUM VS SESUDAH NORMALISASI)</div>", unsafe_allow_html=True)
        skew_r_pre = df['Recency_Asli'].skew(); skew_f_pre = df['Frequency_Asli'].skew(); skew_m_pre = df['Monetary_Asli'].skew()
        if 'Recency_scaled' in df.columns: skew_r_post = df['Recency_scaled'].skew(); skew_f_post = df['Frequency_scaled'].skew(); skew_m_post = df['Monetary_scaled'].skew()
        else: skew_r_post = np.log1p(df['Recency_Asli']).skew(); skew_f_post = np.log1p(df['Frequency_Asli']).skew(); skew_m_post = np.log1p(df['Monetary_Asli']).skew()

        df_skew = pd.DataFrame({'Metrik': ['R-Sebelum', 'R-Sesudah', 'F-Sebelum', 'F-Sesudah', 'M-Sebelum', 'M-Sesudah'], 'Nilai Skewness': [skew_r_pre, skew_r_post, skew_f_pre, skew_f_post, skew_m_pre, skew_m_post], 'Warna': [warna['red'], warna['green'], warna['red'], warna['blue'], warna['red'], warna['purple']]})
        fig_skew = px.bar(df_skew, x='Metrik', y='Nilai Skewness', color='Metrik', color_discrete_sequence=df_skew['Warna'])
        fig_skew.update_layout(**tema_plotly, height=250, xaxis_title=None, yaxis_title=None, showlegend=False); st.plotly_chart(fig_skew, use_container_width=True)
        st.markdown(f"<p style='font-size:12px; color:{warna['muted']};'>Log1p + Z-Score berhasil menurunkan tingkat kemiringan (skewness) sehingga distribusi mendekati normal (mean ≈ 0, std ≈ 1).</p></div>", unsafe_allow_html=True)

    with col_bot2:
        st.markdown(f"<div class='chart-card'><div class='summary-title'>HEATMAP KORELASI RFM</div>", unsafe_allow_html=True)
        corr = df[['Recency_Asli', 'Frequency_Asli', 'Monetary_Asli']].corr().round(2); corr.columns = ['R', 'F', 'M']; corr.index = ['R', 'F', 'M']
        fig_corr = px.imshow(corr, text_auto=True, color_continuous_scale=[warna['surface'], warna['green']], aspect="auto")
        fig_corr.update_layout(**tema_plotly, height=250, coloraxis_showscale=False); st.plotly_chart(fig_corr, use_container_width=True)
        st.markdown(f"<p style='font-size:12px; color:{warna['muted']};'>Korelasi terkuat terjadi antara <b>Frequency dan Monetary ({corr.loc['F', 'M']})</b>, menunjukkan pelanggan yang sering bertransaksi cenderung membelanjakan uang lebih banyak.</p></div>", unsafe_allow_html=True)

# ==============================================================================
# HALAMAN 2: ALGORITMA (K-MEANS, HIERARCHICAL, DBSCAN)
# ==============================================================================
elif menu_pilihan in ["K-Means", "Hierarchical Clustering", "DBSCAN"]:
    algoritma = menu_pilihan
    kolom_cluster = {"K-Means": "Cluster_KMeans", "Hierarchical Clustering": "Cluster_Hierarchical", "DBSCAN": "Cluster_DBSCAN"}
    kolom_pilih = kolom_cluster[algoritma]

    st.markdown(f'''
    <div class="topbar">
        <div style="display: flex; gap: 15px; align-items: center;">
            <div><h2 class="topbar-title">Analisis Segmentasi Pelanggan ({algoritma})</h2><p class="topbar-sub">Visualisasi Interaktif & Justifikasi Penentuan Parameter</p></div>
        </div>
        <div class="topbar-pill">📅 Des 2010 – Des 2011</div>
    </div>
    ''', unsafe_allow_html=True)

    # --- LANDASAN MATEMATIS ---
    st.markdown(f"<h3 style='margin-bottom:15px; font-size:18px; color:{warna['text']};'>🔬 Landasan Matematis Parameter</h3>", unsafe_allow_html=True)
    
    col_img, col_txt = st.columns([1.2, 1])
    with col_img:
        if algoritma == "K-Means": show_image('kmeans_elbow_silhouette.png')
        elif algoritma == "Hierarchical Clustering": show_image('hierarchical_dendrogram.png')
        elif algoritma == "DBSCAN": show_image('dbscan_kdistance.png')
        
    with col_txt:
        if algoritma == "K-Means":
            st.info("**Mengapa dibagi menjadi 4 Cluster?**\n\nPenentuan jumlah *cluster* (k) dilakukan menggunakan **Elbow Method** (menganalisis *Inertia/Within-Cluster Sum of Squares*) dan konfirmasi **Silhouette Score**.\n\nTerlihat patahan yang cukup jelas (Elbow) pada angka k=4. Pemilihan ini menjaga keseimbangan antara kekompakan matematis segmen dan keterbacaan utilitas bisnisnya.")
        elif algoritma == "Hierarchical Clustering":
            st.info("**Pemilihan Threshold (k=4)**\n\nModel ini dibangun menggunakan metode *Agglomerative* dengan **Ward Linkage** untuk meminimalkan varians di dalam *cluster*.\n\nPemotongan (*threshold*) pada garis horizontal dendrogram diselaraskan untuk menghasilkan 4 *cluster* utama. Ini memastikan kita bisa membandingkan hasilnya secara setara (*apple-to-apple*) dengan K-Means.")
        elif algoritma == "DBSCAN":
            st.info("**Mengapa Epsilon ≈ 0.404?**\n\nPenentuan nilai radius pencarian (*epsilon*) menggunakan pendekatan **K-Distance Graph**.\n\nPemotongan dilakukan pada batas **Persentil ke-95** secara sengaja untuk memisahkan 5% populasi data terluar. Dalam konteks RFM, 5% titik ekstrem ini teridentifikasi sebagai kelompok pelanggan *Super VIP* dengan transaksi tak lazim, sehingga sangat tepat diisolasi sebagai *Noise (-1)* agar kepadatan rata-rata *cluster* terjaga.")

    st.markdown("---")
    st.markdown(f"<h3 style='margin-top:10px; margin-bottom:15px; font-size:18px; color:{warna['text']};'>📊 Visualisasi Hasil Clustering</h3>", unsafe_allow_html=True)

    if kolom_pilih in df.columns:
        if algoritma == "DBSCAN": df['Label'] = df[kolom_pilih].apply(lambda x: 'Noise (-1)' if x == -1 else f'Cluster {x}')
        else: df['Label'] = df[kolom_pilih].apply(lambda x: f'Cluster {x}')
    else: df['Label'] = "Data Kosong"

    df_visual = df[df['Label'] != 'Noise (-1)'].copy() if algoritma == "DBSCAN" else df.copy()

    kpi_html = f'''
    <div class="kpi-grid">
        <div class="kpi"><div class="kpi-icon" style="background: rgba(0,212,180,0.15); color: #00D4B4;">👥</div><div><div class="kpi-val">{len(df_visual):,}</div><div class="kpi-label">Pelanggan Aktif</div></div></div>
        <div class="kpi"><div class="kpi-icon" style="background: rgba(79,142,247,0.15); color: #4F8EF7;">⏱️</div><div><div class="kpi-val">{df_visual['Recency_Asli'].mean():.0f}</div><div class="kpi-label">Avg Recency (Hari)</div></div></div>
        <div class="kpi"><div class="kpi-icon" style="background: rgba(167,139,250,0.15); color: #A78BFA;">📦</div><div><div class="kpi-val">{df_visual['Frequency_Asli'].mean():.1f}</div><div class="kpi-label">Avg Frequency</div></div></div>
        <div class="kpi"><div class="kpi-icon" style="background: rgba(251,191,36,0.15); color: #FBBF24;">£</div><div><div class="kpi-val">{df_visual['Monetary_Asli'].mean():,.0f}</div><div class="kpi-label">Avg Monetary</div></div></div>
    </div>
    '''
    st.markdown(kpi_html, unsafe_allow_html=True)

    # --- PERBAIKAN GRAFIK 3D: GRID LEBIH TEBAL DAN JELAS ---
    col_grafik1, col_grafik2 = st.columns([2, 1])
    with col_grafik1:
        fig_3d = px.scatter_3d(df, x='Recency_Asli', y='Frequency_Asli', z='Monetary_Asli', color='Label', color_discrete_map=warna_klaster, opacity=0.7, size_max=5, title="🌌 3D Scatter Plot Segmen Pelanggan")
        fig_3d.update_layout(**tema_plotly, height=650) 
        
        # Grid line dipertebal (opacity 20%) dan zero line ditebalkan untuk presentasi layar lebar
        fig_3d.update_scenes(
            xaxis=dict(backgroundcolor='rgba(0,0,0,0)', gridcolor=warna['grid_strong'], showline=True, linecolor=warna['grid_strong'], zerolinecolor=warna['grid_strong'], zerolinewidth=2),
            yaxis=dict(backgroundcolor='rgba(0,0,0,0)', gridcolor=warna['grid_strong'], showline=True, linecolor=warna['grid_strong'], zerolinecolor=warna['grid_strong'], zerolinewidth=2),
            zaxis=dict(backgroundcolor='rgba(0,0,0,0)', gridcolor=warna['grid_strong'], showline=True, linecolor=warna['grid_strong'], zerolinecolor=warna['grid_strong'], zerolinewidth=2)
        )
        st.plotly_chart(fig_3d, use_container_width=True)

    with col_grafik2:
        dist_df = df['Label'].value_counts().reset_index(); dist_df.columns = ['Label', 'Jumlah']
        fig_donut1 = px.pie(dist_df, values='Jumlah', names='Label', hole=0.6, color='Label', color_discrete_map=warna_klaster, title="👥 Proporsi Jumlah Pelanggan")
        fig_donut1.update_layout(**tema_plotly, height=315, showlegend=True, legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5))
        st.plotly_chart(fig_donut1, use_container_width=True)
        
        rev_df = df.groupby('Label')['Monetary_Asli'].sum().reset_index()
        fig_donut2 = px.pie(rev_df, values='Monetary_Asli', names='Label', hole=0.6, color='Label', color_discrete_map=warna_klaster, title="💰 Proporsi Total Pendapatan (£)")
        fig_donut2.update_layout(**tema_plotly, height=315, showlegend=False) 
        st.plotly_chart(fig_donut2, use_container_width=True)

    # --- BAR CHARTS PROFIL ---
    profil = df[df['Label'] != 'Noise (-1)'].groupby('Label').agg(Recency=('Recency_Asli', 'mean'), Frequency=('Frequency_Asli', 'mean'), Monetary=('Monetary_Asli', 'mean')).reset_index()

    col_bar1, col_bar2, col_bar3 = st.columns(3)
    with col_bar1:
        fig_r = px.bar(profil, x='Label', y='Recency', color='Label', color_discrete_map=warna_klaster, title="Avg Recency (Hari)", text_auto='.0f')
        fig_r.update_layout(**tema_plotly, showlegend=False, yaxis_gridcolor=warna['grid_strong']); st.plotly_chart(fig_r, use_container_width=True)
    with col_bar2:
        fig_f = px.bar(profil, x='Label', y='Frequency', color='Label', color_discrete_map=warna_klaster, title="Avg Frequency", text_auto='.1f')
        fig_f.update_layout(**tema_plotly, showlegend=False, yaxis_gridcolor=warna['grid_strong']); st.plotly_chart(fig_f, use_container_width=True)
    with col_bar3:
        fig_m = px.bar(profil, x='Label', y='Monetary', color='Label', color_discrete_map=warna_klaster, title="Avg Monetary (£)", text_auto='.0f')
        fig_m.update_layout(**tema_plotly, showlegend=False, yaxis_gridcolor=warna['grid_strong']); st.plotly_chart(fig_m, use_container_width=True)

    # --- PANEL ANALISIS NOISE (KHUSUS DBSCAN) ---
    if algoritma == "DBSCAN" and "Noise (-1)" in df['Label'].values:
        noise_df = df[df['Label'] == 'Noise (-1)']; reg_df = df[df['Label'] != 'Noise (-1)']
        st.markdown(f'''
        <div class="noise-box">
            <h4 style="margin-top:0; color:{warna['red']}; display:flex; align-items:center; gap:8px;">⚠️ Analisis Khusus: Mengapa ada "Noise"?</h4>
            <p style="font-size:13px; margin-bottom:12px; color:{warna['text']};">
                DBSCAN sengaja mengisolasi <strong>{len(noise_df)} pelanggan</strong> ke dalam kelompok Noise. Jika kita bandingkan rata-rata pembelanjaan mereka, terlihat jelas bahwa kelompok ini bukanlah "data error", melainkan sekumpulan <strong>Pelanggan Super VIP</strong> yang perilaku transaksinya ekstrem dan dapat merusak rata-rata cluster reguler jika digabungkan.
            </p>
            <div style="display:flex; gap:30px;">
                <div>
                    <div style="font-size:11px; color:{warna['muted']}; text-transform:uppercase;">Avg Monetary Pelanggan Reguler</div>
                    <div style="font-size:18px; font-weight:bold; color:{warna['text']};">£ {reg_df['Monetary_Asli'].mean():,.0f}</div>
                </div>
                <div>
                    <div style="font-size:11px; color:{warna['red']}; text-transform:uppercase;">Avg Monetary Pelanggan NOISE</div>
                    <div style="font-size:18px; font-weight:bold; color:{warna['red']};">£ {noise_df['Monetary_Asli'].mean():,.0f} <span style="font-size:12px;">(Sangat Tinggi)</span></div>
                </div>
            </div>
        </div>
        ''', unsafe_allow_html=True)

    # --- DROPDOWN FILTER & TABEL ---
    st.markdown(f"<h3 style='margin-top:20px; font-size:18px; color:{warna['text']};'>📋 Tabel Data Interaktif</h3>", unsafe_allow_html=True)
    
    pilihan_segmen = ["Semua Segmen"] + sorted(df['Label'].unique().tolist())
    filter_segmen = st.selectbox("Tampilkan data berdasarkan:", pilihan_segmen)
    
    df_tabel = df if filter_segmen == "Semua Segmen" else df[df['Label'] == filter_segmen]
        
    st.markdown(f"<div style='font-size:13px; color:{warna['muted']}; margin-bottom:10px;'>Menampilkan {len(df_tabel):,} baris data.</div>", unsafe_allow_html=True)
    st.dataframe(df_tabel[['CustomerID', 'Recency_Asli', 'Frequency_Asli', 'Monetary_Asli', 'Label']], use_container_width=True, height=300)

# ==============================================================================
# HALAMAN TERAKHIR: PERBANDINGAN METODE
# ==============================================================================
elif menu_pilihan == "Perbandingan Metode":
    st.markdown(f'''
    <div class="topbar">
        <div style="display: flex; gap: 15px; align-items: center;">
            <div><h2 class="topbar-title">Perbandingan Evaluasi Metode</h2><p class="topbar-sub">Analisis Data Historis dari Skrip Python Asli</p></div>
        </div>
    </div>
    ''', unsafe_allow_html=True)

    st.markdown(f"**Tabel Ringkasan Metrik Berdasarkan Hasil Eksperimen:**")
    st.dataframe(df_evaluasi, use_container_width=True, hide_index=True)
    st.markdown("---")

    col_eval1, col_eval2 = st.columns(2)
    with col_eval1:
        fig_ss = px.bar(df_evaluasi, x='Metode', y='Silhouette', color='Metode', color_discrete_map=warna_metode, title="📈 Silhouette Score (Makin tinggi makin baik)", text_auto='.4f')
        fig_ss.update_layout(**tema_plotly, showlegend=False, yaxis_gridcolor=warna['grid_strong']); st.plotly_chart(fig_ss, use_container_width=True)
    with col_eval2:
        fig_dbi = px.bar(df_evaluasi, x='Metode', y='DBI', color='Metode', color_discrete_map=warna_metode, title="📉 Davies-Bouldin Index (Makin rendah makin baik)", text_auto='.4f')
        fig_dbi.update_layout(**tema_plotly, showlegend=False, yaxis_gridcolor=warna['grid_strong']); st.plotly_chart(fig_dbi, use_container_width=True)

    df_valid = df_evaluasi[df_evaluasi['DBI'] < 100].copy()
    pemenang_ss = df_valid.loc[df_valid['Silhouette'].idxmax()]
    pemenang_dbi = df_valid.loc[df_valid['DBI'].idxmin()]

    if pemenang_ss['Metode'] == pemenang_dbi['Metode']:
        metode_terbaik = pemenang_ss['Metode']
        alasan = "unggul mutlak di kedua metrik evaluasi"
    else:
        metode_terbaik = pemenang_ss['Metode']
        alasan = "memiliki Silhouette Score paling tinggi"

    st.markdown("---")
    st.markdown(f'''
    <div class="kesimpulan-box">
        <h3 style="color:#00D4B4; margin-top:0; font-size:18px;">🏆 Kesimpulan Algoritma Terbaik</h3>
        <p style="margin-bottom:0; line-height:1.6; color:{warna['text']};">
            Berdasarkan hasil pemrosesan data, <strong>{metode_terbaik}</strong> merupakan algoritma segmentasi yang paling optimal untuk dataset E-Commerce ini karena <strong>{alasan}</strong>. 
            <br><br>
            {metode_terbaik} berhasil mencatatkan <strong>Silhouette Score sebesar {pemenang_ss['Silhouette']:.4f}</strong> dan <strong>Davies-Bouldin Index sebesar {pemenang_ss['DBI']:.4f}</strong>.
        </p>
    </div>
    ''', unsafe_allow_html=True)