# main.py
import streamlit as st
from supabase import create_client, Client
from dotenv import load_dotenv
import os

# Impor halaman-halaman
from login_page import show_login_page
from chatbot_page import show_chatbot_page
from admin_page import show_admin_page

# --- 1. Konfigurasi Awal ---
st.set_page_config(page_title="Chatbot", layout="wide")
load_dotenv()

# --- 2. Inisialisasi Koneksi Supabase (Cache) ---
@st.cache_resource
def init_supabase():
    """Inisialisasi dan kembalikan Supabase client."""
    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_ANON_KEY")
    
    if not supabase_url or not supabase_key:
        st.error("SUPABASE_URL atau SUPABASE_ANON_KEY tidak ditemukan di .env")
        st.stop()
        
    return create_client(supabase_url, supabase_key)

supabase_client = init_supabase()

# --- 3. Inisialisasi Google API Key ---
google_api_key = os.getenv("GOOGLE_API_KEY")
if not google_api_key:
    st.error("GOOGLE_API_KEY tidak ditemukan di .env")
    st.stop()
st.session_state['google_api_key'] = google_api_key


# --- 4. Inisialisasi Session State ---
if 'supabase' not in st.session_state:
    st.session_state['supabase'] = supabase_client
if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False
if 'username' not in st.session_state:
    st.session_state['username'] = None

# --- 5. Logika Router Utama ---
def main_router():
    if not st.session_state['logged_in']:
        show_login_page()
    else:
        if st.session_state['username'] == 'admin':
            show_admin_page()
        else:
            show_chatbot_page()

if __name__ == "__main__":
    main_router()