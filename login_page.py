# login_page.py
import streamlit as st
import bcrypt

def check_password(password, hashed_password):
    """Mengecek apakah password input cocok dengan hash di DB"""
    try:
        return bcrypt.checkpw(password.encode('utf-8'), hashed_password.encode('utf-8'))
    except ValueError:
        return False

def show_login_page():
    """Menampilkan halaman login."""
    st.set_page_config(page_title="Login")
    st.markdown("<h1 style='text-align: center;'>DIGICHATBOT</h1>", unsafe_allow_html=True)
    st.markdown("<div style='text-align: center;'>Silakan login untuk melanjutkan</div>", unsafe_allow_html=True)
    st.write("---")

    # Ambil koneksi supabase dari session_state
    supabase = st.session_state['supabase']
    
    col1, col2, col3 = st.columns([1, 1.5, 1])
    with col2:
        with st.form("login_form"):
            username = st.text_input("Username", placeholder="Masukkan username Anda")
            password = st.text_input("Password", type="password", placeholder="Masukkan password Anda")
            submitted = st.form_submit_button("Login", use_container_width=True, type="primary")

            if submitted:
                if not username or not password:
                    st.warning("Username dan Password tidak boleh kosong.")
                    return

                with st.spinner("Mencoba login..."):
                    try:
                        # Cari user di database
                        data, count = supabase.table('users').select('*').eq('username', username).execute()
                        
                        if data and len(data[1]) > 0:
                            user_data = data[1][0]
                            hashed_password_db = user_data['hashed_password']
                            
                            # Cek password
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