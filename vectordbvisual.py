import streamlit as st
import pandas as pd
import umap.umap_ as umap
import plotly.express as px
import os
import json # <--- TAMBAHKAN INI
from dotenv import load_dotenv
from supabase.client import Client, create_client
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# --- Konfigurasi Awal ---
load_dotenv()

# Konfigurasi Supabase dan Google API
supabase_url = os.getenv("SUPABASE_URL")
supabase_key = os.getenv("SUPABASE_ANON_KEY")
google_api_key = os.getenv("GOOGLE_API_KEY")

if not all([supabase_url, supabase_key, google_api_key]):
    st.error("Pastikan SUPABASE_URL, SUPABASE_ANON_KEY, dan GOOGLE_API_KEY terisi di file .env Anda.")
    st.stop()

# Inisialisasi Supabase Client
try:
    supabase: Client = create_client(supabase_url, supabase_key)
except Exception as e:
    st.error(f"Gagal menginisialisasi Supabase Client: {e}")
    st.stop()
    
# --- Fungsi Inti Visualisasi ---

@st.cache_data(show_spinner="Mengambil vektor dari Supabase...")
def get_vectors_and_metadata(_supabase_client):
    """Mengambil vektor (embeddings) dan metadata dari Supabase."""
    try:
        # Mengambil kolom 'embedding', 'content' (teks), dan 'metadata'
        response = _supabase_client.table('documents').select('embedding, content, metadata').execute()
        
        data = []
        for item in response.data:
            if item.get('embedding') and item.get('metadata') and item.get('content'):
                
                # FIX KONVERSI: Mengurai string vektor menjadi list float yang valid
                try:
                    # Supabase client mengembalikan vektor sebagai string (e.g., "[0.1, 0.2, ...]")
                    # Kita harus mengubahnya menjadi list float yang dikenali oleh Pandas/UMAP
                    vector_list = json.loads(item['embedding'])
                except (json.JSONDecodeError, TypeError) as e:
                    st.warning(f"Melewatkan data karena gagal mengurai vektor: {e}")
                    continue
                
                text_preview = item['content'][:100].replace('\n', ' ') + "..."
                
                data.append({
                    'vector': vector_list, # <--- SEKARANG ADALAH LIST OF FLOAT
                    'text': text_preview, 
                    'source': item['metadata'].get('source', 'Unknown File'),
                })
        
        return pd.DataFrame(data)
    except Exception as e:
        st.error(f"Gagal mengambil vektor dari Supabase: {e}")
        return pd.DataFrame()

# Pengurangan dimensi dan Plotting di-cache karena merupakan operasi yang intensif
@st.cache_data(show_spinner="Memproyeksikan dimensi (UMAP) dan membuat plot...")
def reduce_and_plot_vectors_3d(df):
    """Mengurangi dimensi (UMAP) dan membuat plot 3D interaktif."""
    
    if df.empty or 'vector' not in df.columns or len(df) < 5:
        st.warning("Data tidak cukup untuk visualisasi 3D (disarankan minimal 5 titik).")
        return None

    # Vektor sekarang adalah list of list of floats
    vectors = df['vector'].tolist()
    
    # UMAP: Proyeksi dimensi tinggi (e.g., 768) ke 3D
    reducer = umap.UMAP(n_components=3, random_state=42, metric='cosine')
    embedding_3d = reducer.fit_transform(vectors)
    
    # Tambahkan koordinat 3D ke DataFrame
    df['x'] = embedding_3d[:, 0]
    df['y'] = embedding_3d[:, 1]
    df['z'] = embedding_3d[:, 2]

    # Membuat Plot 3D Interaktif dengan Plotly
    fig = px.scatter_3d(
        df, 
        x='x', y='y', z='z',
        color='source',
        hover_name='text',
        title=f'Visualisasi Vector Embeddings 3D (N={len(df)})',
        height=750
    )
    
    fig.update_traces(marker=dict(size=4, opacity=0.8))
    fig.update_layout(
        scene=dict(
            xaxis_title='Dimensi 1 (UMAP)', 
            yaxis_title='Dimensi 2 (UMAP)', 
            zaxis_title='Dimensi 3 (UMAP)'
        )
    )
    
    return fig

# --- Tampilan Streamlit (UI) ---

def main_visual():
    st.set_page_config(page_title="Vector DB Visualizer 3D", layout="wide")
    st.markdown("<h1 style='text-align: center;'>Visualisasi 3D Vector Embeddings</h1>", unsafe_allow_html=True)
    st.markdown("<div style='text-align: center;'>Memvisualisasikan kedekatan semantik potongan teks dari Supabase.</div>", unsafe_allow_html=True)
    st.write("---")
    
    if 'run_visualization' not in st.session_state:
        st.session_state.run_visualization = False
    
    with st.sidebar:
        st.header("Kontrol")
        st.info("Visualisasi ini menggunakan UMAP. Titik yang berkelompok erat menunjukkan kesamaan semantik yang tinggi.")
        
        if st.button("🔄 Muat & Visualisasikan Vektor", type="primary"):
            st.session_state.run_visualization = True
        
        st.markdown("---")
        st.subheader("Informasi Database")
        try:
            response = supabase.table('documents').select('id', count='exact').execute()
            st.info(f"Total Chunks di DB: **{response.count}**")
        except Exception:
            st.warning("Tidak dapat terhubung atau menghitung data.")

    if st.session_state.run_visualization:
        
        df = get_vectors_and_metadata(supabase)
        
        if not df.empty and len(df) >= 5:
            
            fig = reduce_and_plot_vectors_3d(df.copy())
            
            if fig:
                st.plotly_chart(fig, use_container_width=True)
                
                st.markdown("#### Interpretasi Plot:")
                st.markdown("- Geser (*drag*) dan Putar plot untuk menjelajahi data.")
                st.markdown("- Arahkan kursor (*hover*) ke titik untuk melihat **potongan teks asli** dan **sumber file**.")
        else:
            if not df.empty:
                st.warning(f"Jumlah vektor hanya {len(df)}. Minimal 5 data diperlukan untuk visualisasi 3D.")
            else:
                st.warning("Tidak ada vektor yang ditemukan di tabel 'documents' Supabase. Pastikan proses dokumen sudah dilakukan.")

if __name__ == "__main__":
    main_visual()