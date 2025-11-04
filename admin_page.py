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
import pandas as pd
import altair as alt

# Impor fungsi chat dari chatbot_page
from chatbot_page import get_conversation_chain 

# --- Fungsi Helper (Auth, PDF) ---

def hash_password(password):
    """Meng-hash password untuk disimpan"""
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

def get_pdf_text(pdf_docs):
    """Mengekstrak teks dari daftar file PDF yang diunggah."""
    text = ""
    for pdf in pdf_docs:
        try:
            pdf.seek(0) # PENTING: Setel ulang pointer file
            pdf_reader = PdfReader(pdf)
            for page in pdf_reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text
        except Exception as e:
            st.warning(f"Tidak dapat membaca file {pdf.name}: {e}")
    return text

def get_text_chunks(text, file_name, classification):
    """Memecah teks mentah menjadi potongan-potongan (chunks) dengan metadata."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_text(text)
    documents = [
        Document(
            page_content=chunk, 
            metadata={
                "source": file_name,
                "classification": classification
            }
        ) 
        for chunk in chunks
    ]
    return documents

def store_in_supabase(documents, pdf_file_object):
    """
    Membuat embedding, menyimpan chunks ke DB, DAN mengunggah file asli ke Storage.
    Mengembalikan True jika sukses, False jika gagal.
    """
    if not documents:
        st.warning("Tidak ada teks yang dapat diproses.")
        return False
    
    supabase = st.session_state['supabase']
    google_api_key = st.session_state['google_api_key']
    file_name = documents[0].metadata['source'] # Ini sekarang nama yang sudah bersih

    try:
        # 1. Unggah file PDF asli ke Supabase Storage
        pdf_file_object.seek(0)
        file_content = pdf_file_object.read()
        
        try:
            # Gunakan .from_() sesuai v2 supabase-py
            supabase.storage.from_('pdf_documents').upload(
                path=file_name,
                file=file_content,
                file_options={"content-type": "application/pdf"}
            )
        except Exception as storage_error:
            if "Duplicate" in str(storage_error) or "409" in str(storage_error):
                st.warning(f"File '{file_name}' sudah ada di Storage. Melanjutkan proses RAG.")
            else:
                raise storage_error 

        # 2. Simpan text chunks dan embeddings ke Database
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
        st.success(f"File '{file_name}' berhasil diproses (RAG dan Storage)!")
        return True # Sukses
    
    except Exception as e:
        st.error(f"GAGAL memproses '{file_name}': {e}")
        return False # Gagal

# --- Fungsi Helper (Ambil Data) ---

def get_document_list(supabase):
    """Mengambil daftar dokumen dan metadatanya."""
    try:
        response = supabase.table('documents').select('metadata').execute()
        if response.data:
            unique_docs = {item['metadata']['source']: item['metadata'] for item in response.data}.values()
            return list(unique_docs)
        return []
    except Exception as e:
        st.error(f"Gagal mengambil daftar file dari Supabase: {e}")
        return []

def delete_document_from_supabase(filename):
    """Menghapus entri dari DB DAN file dari Storage."""
    supabase = st.session_state['supabase']
    try:
        supabase.table('documents').delete().eq('metadata->>source', filename).execute()
        
        # Gunakan .from_()
        supabase.storage.from_('pdf_documents').remove([filename])
        
        return True, f"Dokumen '{filename}' berhasil dihapus dari DB dan Storage."
    except Exception as e:
        if "No such object" in str(e):
             return True, f"Dokumen '{filename}' berhasil dihapus dari DB (tidak ditemukan di Storage)."
        return False, f"Gagal menghapus dokumen: {e}"

# --- FUNGSI Helper Dashboard ---
# (Tidak ada perubahan di sini)
@st.cache_data(ttl=600) 
def get_dashboard_data(_supabase):
    try:
        users_response = _supabase.table('users').select('username', 'created_at').execute()
        users_data = users_response.data or []
        docs_response = _supabase.table('documents').select('metadata').execute()
        docs_data = docs_response.data or []
        return users_data, docs_data
    except Exception as e:
        st.error(f"Gagal memuat data dashboard: {e}")
        return [], []

def create_user_chart(users_data):
    if not users_data:
        return st.info("Belum ada data pengguna.")
    df = pd.DataFrame(users_data)
    df['created_at'] = pd.to_datetime(df['created_at'])
    df_monthly = df.set_index('created_at').resample('M').count()['username'].reset_index()
    df_monthly['created_at'] = df_monthly['created_at'].dt.strftime('%Y-%m')
    chart = alt.Chart(df_monthly).mark_bar().encode(
        x=alt.X('created_at', title='Bulan'),
        y=alt.Y('username', title='Jumlah Pendaftaran')
    ).properties(title="Pendaftaran Pengguna Baru per Bulan")
    st.altair_chart(chart, use_container_width=True)

def create_doc_chart(docs_data):
    if not docs_data:
        return st.info("Belum ada data dokumen.")
    unique_docs = {}
    for item in docs_data:
        source = item['metadata'].get('source', 'N/A')
        classification = item['metadata'].get('classification', 'Lain-lain')
        unique_docs[source] = classification
    df_source = pd.DataFrame(list(unique_docs.values()), columns=['classification'])
    df_counts = df_source['classification'].value_counts().reset_index()
    df_counts.columns = ['classification', 'count']
    base = alt.Chart(df_counts).encode(
       theta=alt.Theta("count:Q", stack=True)
    ).properties(title="Klasifikasi Dokumen")
    pie = base.mark_arc(outerRadius=120).encode(
        color=alt.Color("classification:N"),
        order=alt.Order("count:Q", sort="descending")
    )
    text = base.mark_text(radius=140).encode(
        text=alt.Text("count:Q"),
        order=alt.Order("count:Q", sort="descending"),
        color=alt.value("black")
    )
    st.altair_chart(pie + text, use_container_width=True)


def init_admin_chat_session():
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
    if 'file_to_delete' not in st.session_state:
        st.session_state.file_to_delete = None

# --- Tampilan Utama Halaman Admin ---
# (Tidak ada perubahan di sini)
def show_admin_page():
    """Menampilkan halaman admin dengan panel manajemen di area utama."""
    
    init_admin_chat_session()
    supabase = st.session_state['supabase']

    # --- Sidebar Navigasi ---
    with st.sidebar:
        st.markdown(f"### Admin Panel, {st.session_state['username']}!")
        st.write("---")
        st.markdown("""
        <a href="#dashboard" style="text-decoration: none; color: inherit;"><div style="padding: 10px; border-radius: 5px; margin-bottom: 5px; background-color: #262730;">📊 Dashboard</div></a>
        <a href="#manajemen-dokumen" style="text-decoration: none; color: inherit;"><div style="padding: 10px; border-radius: 5px; margin-bottom: 5px; background-color: #262730;">📄 Manajemen Dokumen</div></a>
        <a href="#manajemen-pengguna" style="text-decoration: none; color: inherit;"><div style="padding: 10px; border-radius: 5px; margin-bottom: 5px; background-color: #262730;">👤 Manajemen Pengguna</div></a>
        """, unsafe_allow_html=True)
        st.markdown("<div style='flex-grow: 1;'></div>", unsafe_allow_html=True)
        if st.button("Log Out", use_container_width=True):
            st.session_state.logged_in = False
            st.session_state.username = None
            st.session_state.chat_history = []
            st.session_state.conversation_chain = None
            st.session_state.file_to_delete = None
            st.rerun()

    # --- Judul Utama ---
    st.markdown("<h1 style='text-align: center;'>DIGICHATBOT (ADMIN PANEL)</h1>", unsafe_allow_html=True)
    st.write("---")
    
    # --- 1. Bagian Dashboard ---
    st.markdown('<h2 id="dashboard">📊 Dashboard</h2>', unsafe_allow_html=True)
    users_data, docs_data = get_dashboard_data(supabase)
    col_dash_1, col_dash_2 = st.columns([1, 2])
    with col_dash_1:
        st.metric("Total Pengguna", len(users_data))
        st.metric("Total Dokumen", len(set(item['metadata'].get('source') for item in docs_data)))
    with col_dash_2:
        create_doc_chart(docs_data)
    st.markdown("---")
    create_user_chart(users_data)
    st.divider()

    # --- 2. Bagian Manajemen Dokumen ---
    st.markdown('<h2 id="manajemen-dokumen">📄 Manajemen Dokumen</h2>', unsafe_allow_html=True)
    st.subheader("Unggah Dokumen Baru")
    
    classification_options = ["Pilih Klasifikasi...", "Dok. Universitas", "Dok. Fakultas", "Dok. Prodi"]
    classification = st.selectbox("Klasifikasi Dokumen", options=classification_options)
    
    pdf_docs = st.file_uploader(
        "Pilih file PDF untuk diunggah", 
        accept_multiple_files=True,
        type="pdf"
    )
    
    if st.button("Proses Dokumen", use_container_width=True, type="primary"):
        if not pdf_docs:
            st.warning("Silakan unggah setidaknya satu file PDF.")
        elif classification == "Pilih Klasifikasi...":
            st.warning("Silakan pilih klasifikasi dokumen.")
        else:
            with st.spinner("Memproses file..."):
                all_docs_list = [item['metadata'].get('source') for item in get_document_list(supabase)]
                all_successful = True 
                
                for pdf in pdf_docs:
                    # --- PERBAIKAN DI SINI ---
                    # 1. Bersihkan nama file
                    sanitized_name = pdf.name.replace(" ", "_")
                    
                    # 2. Cek nama yang sudah bersih
                    if sanitized_name in all_docs_list:
                        st.warning(f"File '{sanitized_name}' sudah ada. Proses dilewati.")
                        continue
                    
                    raw_text = get_pdf_text([pdf])
                    if raw_text:
                        # 3. Kirim nama yang sudah bersih
                        text_chunks = get_text_chunks(raw_text, sanitized_name, classification)
                        success = store_in_supabase(text_chunks, pdf) 
                        if not success:
                            all_successful = False
                    else:
                        st.warning(f"Tidak dapat mengekstrak teks dari '{pdf.name}'.")
                        all_successful = False
                
                if all_successful:
                    st.rerun()
                else:
                    st.error("Satu atau lebih file gagal diproses. Pesan error ada di atas. Halaman tidak di-refresh.")
    
    st.markdown("---")
    st.subheader("Database Dokumen Saat Ini")

    if st.session_state.file_to_delete:
        with st.container():
            file_name = st.session_state.file_to_delete
            st.warning(f"**Konfirmasi Penghapusan**\n\nYakin ingin menghapus **'{file_name}'**? Ini akan menghapus semua data terkait.", icon="⚠️")
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

    document_list = get_document_list(supabase)
    if document_list:
        with st.container(height=400): 
            for meta in document_list:
                doc_name = meta.get('source', 'Nama Tidak Ditemukan')
                doc_class = meta.get('classification', 'Belum Terklasifikasi')
                col1, col2 = st.columns([4, 1])
                with col1:
                    st.info(f"📄 **{doc_name}**\n\n*Klasifikasi: {doc_class}*")
                with col2:
                    is_modal_active = st.session_state.file_to_delete is not None
                    if st.button("🗑️", key=f"delete_{doc_name}", help=f"Hapus {doc_name}", use_container_width=True, disabled=is_modal_active):
                        st.session_state.file_to_delete = doc_name
                        st.rerun()
    else:
        st.info("Belum ada dokumen di database.")

    st.divider()

    # --- 3. Bagian Manajemen Pengguna ---
    st.markdown('<h2 id="manajemen-pengguna">👤 Manajemen Pengguna</h2>', unsafe_allow_html=True)
    col_user_1, col_user_2 = st.columns(2)
    with col_user_1:
        st.subheader("Tambah Pengguna Baru")
        with st.form("admin_add_user"):
            new_username = st.text_input("Username Baru")
            new_password = st.text_input("Password Baru", type="password")
            submit_admin = st.form_submit_button("Tambah Pengguna", use_container_width=True, type="primary")
            if submit_admin:
                if new_username and new_password:
                    if new_username == 'admin':
                        st.error("Tidak dapat membuat pengguna dengan nama 'admin'.")
                    else:
                        try:
                            data, count = supabase.table('users').select('username').eq('username', new_username).execute()
                            if data and len(data[1]) > 0:
                                st.warning(f"Username '{new_username}' sudah ada.")
                            else:
                                hashed_pass = hash_password(new_password)
                                supabase.table('users').insert({"username": new_username, "hashed_password": hashed_pass}).execute()
                                st.success(f"Pengguna '{new_username}' berhasil ditambahkan!")
                                st.rerun()
                        except Exception as e:
                            st.error(f"Gagal menambahkan pengguna: {e}")
                else:
                    st.warning("Username dan password tidak boleh kosong.")
    with col_user_2:
        st.subheader("Daftar Pengguna Saat Ini")
        if users_data:
            with st.container(height=400):
                for user in users_data:
                    if user['username'] != 'admin':
                        st.info(f"👤 {user['username']}")
        else:
            st.info("Hanya 'admin' yang ada.")

    st.divider()

    # --- 4. Expander Test Chatbot ---
    with st.expander("🤖 Test Chatbot (Admin)", expanded=False):
        chat_container = st.container(height=500)
        with chat_container:
            for message in st.session_state.chat_history:
                if isinstance(message, HumanMessage):
                    with st.chat_message("user"):
                        st.markdown(message.content)
                elif isinstance(message, AIMessage):
                    with st.chat_message("assistant"):
                        st.markdown(message.content)

        user_question = st.chat_input("Ajukan pertanyaan disini...")
        if user_question:
            with st.chat_message("user"):
                st.markdown(user_question)
            with st.spinner("DIGICHATBOT sedang memproses..."):
                response = st.session_state.conversation_chain({'question': user_question})
                st.session_state.chat_history = response['chat_history']
                st.rerun()