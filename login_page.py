# login_page.py
import streamlit as st
import bcrypt
import base64 
import os     

def check_password(password, hashed_password):
    """Mengecek apakah password input cocok dengan hash di DB"""
    try:
        return bcrypt.checkpw(password.encode('utf-8'), hashed_password.encode('utf-8'))
    except ValueError:
        return False

# --- FUNGSI HELPER UNTUK MENGUBAH GAMBAR KE BASE64 ---
def get_base64_of_bin_file(bin_file):
    """Membaca file biner dan mengembalikannya sebagai string base64"""
    if not os.path.exists(bin_file):
        return None
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()
# --- AKHIR FUNGSI HELPER ---

def show_login_page():
    """Menampilkan halaman login."""
    st.set_page_config(page_title="Login")

    # --- MEMBUAT STRING BASE64 DARI GAMBAR ---
    img_base64 = get_base64_of_bin_file("login_bg.png")
    
    if img_base64 is None:
        st.error("Error: File 'login_bg.png' tidak ditemukan. Pastikan file ada di folder yang sama.")
        img_base64_string = ""
        gradient = "linear-gradient(rgba(0, 0, 0, 1), rgba(0, 0, 0, 1))"
    else:
        img_base64_string = f'data:image/png;base64,{img_base64}'
        gradient = f'linear-gradient(rgba(0, 0, 0, 0.7), rgba(0, 0, 0, 0.7)), url("{img_base64_string}")'

    
    # --- CSS KUSTOM UNTUK BACKGROUND DAN STYLING ---
    page_styling = f"""
    <style>
    /* Target container utama Streamlit */
    [data-testid="stAppViewContainer"] {{
        /*
         * Kita gunakan f-string Python untuk menyuntikkan
         * string base64 ke dalam CSS
         */
        background-image: {gradient};
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
    }}

    /* Target form login */
    [data-testid="stForm"] {{
        background-color: #1a1a1a;
        padding: 25px;
        border-radius: 10px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.5);
    }}

    /* Opsi: Ubah warna placeholder */
    .stTextInput input::placeholder {{
        color: #999;
    }}
    
    h1, div[style*='text-align: center'] {{
        color: white !important;
    }}
    </style>
    """
    st.markdown(page_styling, unsafe_allow_html=True)

    st.markdown("<h1 style='text-align: center;'>DIGICHATBOT</h1>", unsafe_allow_html=True)
    st.markdown("<div style='text-align: center;'>Silakan login untuk melanjutkan</div>", unsafe_allow_html=True)

    if 'supabase' not in st.session_state:
        st.error("Koneksi Supabase tidak ditemukan di session state.")
        st.stop()
        
    supabase = st.session_state['supabase']
    

    col1, col2, col3 = st.columns([1.5, 1, 1.5])
    
    with col2:
        with st.form("login_form"):
            username = st.text_input("Username", placeholder="Masukkan username Anda", label_visibility="collapsed")
            password = st.text_input("Password", type="password", placeholder="Masukkan password Anda", label_visibility="collapsed")
            submitted = st.form_submit_button("Login", use_container_width=True, type="primary")

            if submitted:
                if not username or not password:
                    st.warning("Username dan Password tidak boleh kosong.")
                else:
                    with st.spinner("Mencoba login..."):
                        try:
                            data, count = supabase.table('users').select('*').eq('username', username).execute()
                            
                            if data and len(data[1]) > 0:
                                user_data = data[1][0]
                                hashed_password_db = user_data['hashed_password']
                                
                                if check_password(password, hashed_password_db):
                                    st.session_state['logged_in'] = True
                                    st.session_state['username'] = user_data['username']
                                    st.rerun()
                                else:
                                    st.error("Username atau password salah")
                            else:
                                st.error("Username atau password salah")
                        except Exception as e:
                            st.error(f"Terjadi kesalahan: {e}")