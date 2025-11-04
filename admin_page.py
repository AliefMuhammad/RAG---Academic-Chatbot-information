# admin_page.py
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
import google.generativeai as genai
from langchain_community.vectorstores import SupabaseVectorStore
from langchain.docstore.document import Document
from langchain_core.messages import AIMessage, HumanMessage
import bcrypt

# Impor fungsi chat dari chatbot_page (DRY - Don't Repeat Yourself)
from chatbot_page import get_conversation_chain 

# --- Fungsi Helper (Khusus Admin) ---

def hash_password(password):
    """Meng-hash password untuk disimpan"""
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

def get_pdf_text(pdf_docs):
    """Mengekstrak teks dari daftar file PDF yang diunggah."""
    text = ""
    for pdf in pdf_docs:
        try:
            pdf_reader = PdfReader(pdf)
            for page in pdf_reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text
        except Exception as e:
            st.warning(f"Tidak dapat membaca file {pdf.name}: {e}")
    return text

def get_text_chunks(text, file_name):
    """Memecah teks mentah menjadi potongan-potongan (chunks) dengan metadata."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_text(text)
    documents = [Document(page_content=chunk, metadata={"source": file_name}) for chunk in chunks]
    return documents

def store_in_supabase(documents):
    """Membuat embedding dan menyimpan text chunks ke Supabase."""
    if not documents:
        st.warning("Tidak ada teks yang dapat diproses.")
        return
    
    supabase = st.session_state['supabase']
    google_api_key = st.session_state['google_api_key']
    
    try:
        genai.configure(api_key=google_api_key)
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=google_api_key)
        SupabaseVectorStore.from_documents(
            documents=documents,
            embedding=embeddings,
            client=supabase,
            table_name="documents",
            query_name="match_documents",
            chunk_size=500,
        )
        st.success(f"File '{documents[0].metadata['source']}' berhasil diproses dan disimpan!")
    except Exception as e:
        st.error(f"Gagal menyimpan ke Supabase: {e}")

def get_uploaded_filenames():
    """Mengambil nama file unik yang sudah ada di Supabase."""
    supabase = st.session_state['supabase']
    try:
        response = supabase.table('documents').select('metadata', count='exact').execute()
        filenames = {item['metadata']['source'] for item in response.data if 'source' in item['metadata']}
        return sorted(list(filenames))
    except Exception as e:
        st.error(f"Gagal mengambil daftar file dari Supabase: {e}")
        return []

def delete_document_from_supabase(filename):
    """Menghapus semua entri yang terkait dengan nama file tertentu dari Supabase."""
    supabase = st.session_state['supabase']
    try:
        supabase.table('documents').delete().eq('metadata->>source', filename).execute()
        return True, f"Dokumen '{filename}' dan datanya berhasil dihapus."
    except Exception as e:
        return False, f"Gagal menghapus dokumen dari Supabase: {e}"

def init_admin_chat_session():
    """Inisialisasi session state yang diperlukan untuk halaman admin."""
    # Inisialisasi chat (jika belum ada)
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'conversation_chain' not in st.session_state:
        supabase = st.session_state['supabase']
        google_api_key = st.session_state['google_api_key']
        
        genai.configure(api_key=google_api_key)
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=google_api_key)
        
        vector_store = SupabaseVectorStore(
            client=supabase,
            embedding=embeddings,
            table_name="documents",
            query_name="match_documents"
        )
        st.session_state.conversation_chain = get_conversation_chain(vector_store, google_api_key)
    
    # Inisialisasi state untuk hapus file
    if 'file_to_delete' not in st.session_state:
        st.session_state.file_to_delete = None

# --- Tampilan Utama Halaman Admin ---

def show_admin_page():
    """Menampilkan halaman admin dengan chat dan sidebar manajemen."""
    st.set_page_config(page_title="Admin - Chatbot", layout="wide")
    
    # Panggil inisialisasi
    init_admin_chat_session()
    supabase = st.session_state['supabase'] # Ambil client supabase

    # --- Tombol Logout (di atas, di luar sidebar) ---
    with st.container():
        col1, col2 = st.columns([10, 1])
        with col1:
            st.markdown(f"<h3 style='margin-bottom: 0;'>Selamat Datang, Admin {st.session_state['username']}!</h3>", unsafe_allow_html=True)
        with col2:
            if st.button("Log Out", use_container_width=True):
                st.session_state.logged_in = False
                st.session_state.username = None
                st.session_state.chat_history = []
                st.session_state.conversation_chain = None
                st.session_state.file_to_delete = None
                st.rerun()

    st.markdown("<h1 style='text-align: center;'>DIGICHATBOT (ADMIN VIEW)</h1>", unsafe_allow_html=True)
    st.write("---")

    # --- Pop up Konfirmasi Hapus ---
    if st.session_state.file_to_delete:
        with st.container():
            file_name = st.session_state.file_to_delete
            st.warning(f"**Konfirmasi Penghapusan**\n\nAnda yakin ingin menghapus dokumen **'{file_name}'** secara permanen dari database? Tindakan ini tidak dapat dibatalkan.", icon="⚠️")
            col1, col2, _ = st.columns([1, 1, 5])
            if col1.button("✅ Ya, Hapus", use_container_width=True, type="primary"):
                with st.spinner(f"Menghapus '{file_name}'..."):
                    success, message = delete_document_from_supabase(file_name)
                    st.toast(message, icon="✅" if success else "❌")
                    st.session_state.file_to_delete = None 
                    st.rerun() 
            if col2.button("❌ Batal", use_container_width=True):
                st.session_state.file_to_delete = None
                st.rerun()

    # --- Tampilan ui Chat (Main Area) ---
    for message in st.session_state.chat_history:
        if isinstance(message, HumanMessage):
            with st.chat_message("user", avatar="🧑‍💻"):
                st.markdown(message.content)
        elif isinstance(message, AIMessage):
            with st.chat_message("assistant", avatar="🤖"):
                st.markdown(message.content)

    # Input
    user_question = st.chat_input("Ajukan pertanyaan disini...")
    if user_question:
        with st.spinner("Memproses..."):
            response = st.session_state.conversation_chain({'question': user_question})
            st.session_state.chat_history = response['chat_history']
            st.rerun()

    # --- Sidebar (Panel Admin) ---
    with st.sidebar:
        st.title("Admin Panel")
        st.markdown("---")

        # --- 1. Manajemen Pengguna ---
        with st.expander("👤 Manajemen Pengguna", expanded=False):
            st.subheader("Tambah Pengguna Baru")
            with st.form("admin_add_user"):
                new_username = st.text_input("Username Baru")
                new_password = st.text_input("Password Baru", type="password")
                submit_admin = st.form_submit_button("Tambah Pengguna")

                if submit_admin:
                    if new_username and new_password:
                        try:
                            # Cek dulu apakah username sudah ada
                            data, count = supabase.table('users').select('username').eq('username', new_username).execute()
                            if data and len(data[1]) > 0:
                                st.warning(f"Username '{new_username}' sudah ada.")
                            else:
                                # Jika belum ada, hash passwordnya dan masukkan
                                hashed_pass = hash_password(new_password)
                                data, count = supabase.table('users').insert({
                                    "username": new_username,
                                    "hashed_password": hashed_pass
                                }).execute()
                                st.success(f"Pengguna '{new_username}' berhasil ditambahkan!")
                        except Exception as e:
                            st.error(f"Gagal menambahkan pengguna: {e}")
                    else:
                        st.warning("Username dan password tidak boleh kosong.")

        st.markdown("---")

        # --- 2. Manajemen Dokumen ---
        st.subheader("Database Dokumen")
        uploaded_files = get_uploaded_filenames()
        if uploaded_files:
            for filename in uploaded_files:
                col1, col2 = st.columns([4, 1])
                with col1:
                    st.info(f"📄 {filename}")
                with col2:
                    if st.button("🗑️", key=f"delete_{filename}", help=f"Hapus {filename}"):
                        st.session_state.file_to_delete = filename
                        st.rerun()
        else:
            st.info("Belum ada dokumen di database.")
        
        st.markdown("---")
        
        st.subheader("Unggah Dokumen Baru")
        pdf_docs = st.file_uploader(
            "Pilih file PDF", 
            accept_multiple_files=True,
            type="pdf"
        )
        
        if st.button("Proses Dokumen"):
            if pdf_docs:
                with st.spinner("Memproses file..."):
                    existing_files = get_uploaded_filenames()
                    for pdf in pdf_docs:
                        if pdf.name in existing_files:
                            st.warning(f"File '{pdf.name}' sudah ada. Proses dilewati.")
                            continue
                        raw_text = get_pdf_text([pdf])
                        if raw_text:
                            text_chunks = get_text_chunks(raw_text, pdf.name)
                            store_in_supabase(text_chunks)
                        else:
                            st.warning(f"Tidak dapat mengekstrak teks dari '{pdf.name}'.")
                    st.rerun()
            else:
                st.warning("Silakan unggah setidaknya satu file PDF.")