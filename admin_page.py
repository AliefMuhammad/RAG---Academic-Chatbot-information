# admin_page.py (Perbaikan Chunking dengan strategy="by_title")
import streamlit as st
import tempfile
import os
import logging # <-- Import untuk membungkam warning

# --- Import Kunci ---
from unstructured.partition.pdf import partition_pdf
from unstructured.cleaners.core import clean_extra_whitespace, clean_non_ascii_chars
import hashlib
import uuid

# --- Hapus import yang tidak perlu ---
# from langchain_text_splitters import RecursiveCharacterTextSplitter # <-- HAPUS INI

from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
import google.generativeai as genai
from langchain_community.vectorstores import SupabaseVectorStore
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage
import bcrypt
import pandas as pd
import altair as alt

# Impor fungsi chat dari chatbot_page
from chatbot_page import get_conversation_chain

# --- Sembunyikan Warning 'pdfminer' (dari log Anda) ---
logging.getLogger("pdfminer").setLevel(logging.ERROR)

# --- Fungsi Helper (Auth) ---
def hash_password(password):
    """Meng-hash password untuk disimpan"""
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

# --- [FUNGSI DIPERBAIKI] Fungsi Akuisisi Data ---

def process_and_store_document(pdf_file_object, file_content, file_name, classification, file_hash):
    """
    Fungsi master untuk memproses satu PDF:
    1. Parsing "hi-res" DENGAN chunking "by_title"
    2. Membuat parent record di DB
    3. Mengunggah file ke Storage
    4. Membuat chunks/documents DARI ELEMEN YANG SUDAH DI-CHUNK
    5. Menyimpan chunks & embeddings ke vector store
    """
    supabase = st.session_state['supabase']
    google_api_key = st.session_state['google_api_key']
    parent_id = str(uuid.uuid4()) # ID unik untuk file induk ini

    try:
        # --- Langkah 1: Parsing PDF dengan Unstructured (STRATEGI BARU) ---
        st.write(f"Memulai 'unstructured' parsing untuk {file_name}...")
        elements = partition_pdf(
            file=pdf_file_object,
            strategy="hi_res", 
            infer_table_structure=True,
            ocr_languages="ind+eng", # Tetap gunakan ini
            extract_images_in_pdf=False,
            
            # --- PERBAIKAN UTAMA: CHUNKING CERDAS ---
            chunking_strategy="by_title",
            max_characters=4000,
            combine_text_under_n_chars=2000,
            new_after_n_chars=3800,
            # --- AKHIR PERBAIKAN ---
        )
        st.write(f"Parsing selesai. Ditemukan {len(elements)} CHUNKS logis.")

        # --- Langkah 2: Buat Dokumen LangChain dari Elemen/Chunks ---
        documents = []
        for el in elements:
            element_type = el.category
            page_number = el.metadata.page_number
            
            # Teks sekarang sudah di-chunk dengan baik (bukan per kata)
            if element_type == "Table":
                text = el.metadata.text_as_html
            else:
                text = str(el)

            text = clean_non_ascii_chars(text)
            text = clean_extra_whitespace(text)
            
            if not text.strip():
                continue 

            # Kita tidak perlu text_splitter lagi, 'el' sudah merupakan chunk
            new_metadata = {
                "source": file_name,
                "classification": classification,
                "file_hash": file_hash,
                "parent_file_id": parent_id,
                "element_type": element_type,
                "page_number": int(page_number) if page_number else 1
            }
            documents.append(Document(page_content=text, metadata=new_metadata))

        if not documents:
            return False, f"Tidak ada teks yang bisa diekstrak dari {file_name}."

        st.write(f"Membuat {len(documents)} vektor dari {file_name}...")

        # --- Langkah 3: Simpan ke Database (Atomik) ---
        
        # 3a. Buat entri di tabel file induk (parent_files)
        supabase.table('parent_files').insert({
            "id": parent_id,
            "file_name": file_name,
            "file_hash": file_hash,
            "classification": classification
        }).execute()
        
        # 3b. Unggah file PDF asli ke Supabase Storage
        try:
            supabase.storage.from_('pdf_documents').upload(
                path=file_name,
                file=file_content,
                file_options={"content-type": "application/pdf", "upsert": "true"}
            )
        except Exception as storage_error:
            if "Duplicate" not in str(storage_error) and "409" not in str(storage_error):
                raise storage_error 

        # 3c. Simpan text chunks dan embeddings ke Database (tabel documents)
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
        return True, f"Sukses memproses {file_name}"

    except Exception as e:
        try:
            supabase.table('parent_files').delete().eq('id', parent_id).execute()
        except Exception as rollback_e:
            st.error(f"Error saat proses: {e}. Gagal rollback parent_files: {rollback_e}")
            
        st.error(f"GAGAL TOTAL memproses '{file_name}': {e}")
        return False, str(e)


# --- [FUNGSI MODIFIKASI] Menggunakan DB Struktur Baru ---
# (Sisa kode di bawah ini TIDAK BERUBAH)

@st.cache_data(ttl=60)
def get_existing_file_hashes(_supabase):
    """[BARU] Mengambil semua hash file yang sudah ada dari tabel parent_files."""
    try:
        response = _supabase.table('parent_files').select('file_hash').execute()
        if response.data:
            return set(item['file_hash'] for item in response.data)
        return set()
    except Exception as e:
        st.error(f"Gagal mengambil hash file: {e}")
        return set()

@st.cache_data(ttl=60)
def get_document_list(_supabase):
    """[MODIFIKASI] Mengambil daftar dokumen dari tabel 'parent_files'."""
    try:
        response = _supabase.table('parent_files').select('file_name, classification').order('created_at', desc=True).execute()
        if response.data:
            return [{"source": item['file_name'], "classification": item['classification']} for item in response.data]
        return []
    except Exception as e:
        st.error(f"Gagal mengambil daftar file dari Supabase: {e}")
        return []

def delete_document_from_supabase(filename):
    """[MODIFIKASI] Menghapus file dari DB (parent & chunks) DAN Storage."""
    supabase = st.session_state['supabase']
    try:
        parent_response = supabase.table('parent_files').select('id').eq('file_name', filename).execute()
        if not parent_response.data:
            return False, f"File '{filename}' tidak ditemukan di tabel parent_files."
        
        parent_id = parent_response.data[0]['id']

        st.write(f"Menghapus chunks untuk parent_id: {parent_id}...")
        supabase.table('documents').delete().eq('metadata->>parent_file_id', parent_id).execute()
        
        st.write(f"Menghapus parent record untuk {filename}...")
        supabase.table('parent_files').delete().eq('id', parent_id).execute()
        
        st.write(f"Menghapus file dari Storage...")
        supabase.storage.from_('pdf_documents').remove([filename])
        
        return True, f"Dokumen '{filename}' (dan semua chunks-nya) berhasil dihapus."
    except Exception as e:
        if "No such object" in str(e):
             return True, f"Dokumen '{filename}' berhasil dihapus dari DB (tidak ditemukan di Storage)."
        return False, f"Gagal menghapus dokumen: {e}"

# --- FUNGSI Helper Dashboard (MODIFIKASI) ---
@st.cache_data(ttl=600) 
def get_dashboard_data(_supabase):
    try:
        users_response = _supabase.table('users').select('username, created_at').execute()
        users_data = users_response.data or []
        docs_response = _supabase.table('parent_files').select('file_name, classification').execute()
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
    df_monthly = df.set_index('created_at').resample('ME').count()['username'].reset_index() # Menggunakan ME
    df_monthly['created_at'] = df_monthly['created_at'].dt.strftime('%Y-%m')
    chart = alt.Chart(df_monthly).mark_bar().encode(
        x=alt.X('created_at', title='Bulan'),
        y=alt.Y('username', title='Jumlah Pendaftaran')
    ).properties(title="Pendaftaran Pengguna Baru per Bulan")
    st.altair_chart(chart, use_container_width=True)


def create_doc_chart(docs_data):
    if not docs_data:
        return st.info("Belum ada data dokumen.")
        
    df_source = pd.DataFrame(docs_data)
    
    if 'classification' not in df_source.columns:
        df_source['classification'] = 'Lain-lain'
    else:
        df_source['classification'] = df_source['classification'].fillna('Lain-lain')

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

# --- Tampilan Utama Halaman Admin (MODIFIKASI LOGIKA UPLOAD) ---
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
        st.metric("Total Dokumen", len(docs_data))
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
            with st.spinner("Memproses file... Ini mungkin butuh waktu lama..."):
                existing_hashes = get_existing_file_hashes(supabase)
                all_successful = True 
                
                for pdf in pdf_docs:
                    sanitized_name = pdf.name.replace(" ", "_")
                    st.markdown(f"--- \n ### Memproses: {sanitized_name}")
                    
                    pdf.seek(0)
                    file_content = pdf.read()
                    file_hash = hashlib.md5(file_content).hexdigest()
                    
                    if file_hash in existing_hashes:
                        st.warning(f"File '{sanitized_name}' (hash: {file_hash[:7]}...) sudah ada di database. Proses dilewati.")
                        continue
                        
                    pdf.seek(0)
                    
                    # --- PERBAIKAN LOGIKA INTI ---
                    # Memanggil fungsi baru (process_and_store_document)
                    # yang menggunakan 'partition_pdf' dengan 'by_title'
                    success, message = process_and_store_document(
                        pdf, 
                        file_content, 
                        sanitized_name, 
                        classification, 
                        file_hash
                    )
                    
                    if not success:
                        all_successful = False
                        st.error(f"Gagal memproses {sanitized_name}: {message}")
                    else:
                        existing_hashes.add(file_hash)

            if all_successful:
                st.success("Semua file baru berhasil diproses!")
                st.cache_data.clear()
                st.rerun()
            else:
                st.error("Satu atau lebih file gagal diproses. Lihat pesan error di atas.")
                st.cache_data.clear()
                st.rerun()
    
    st.markdown("---")
    st.subheader("Database Dokumen Saat Ini")

    # --- Bagian Konfirmasi Hapus ---
    if st.session_state.file_to_delete:
        with st.container():
            file_name = st.session_state.file_to_delete
            st.warning(f"**Konfirmasi Penghapusan**\n\nYakin ingin menghapus **'{file_name}'**? Ini akan menghapus file dari Storage DAN semua data vektor terkait.", icon="⚠️")
            col1, col2, _ = st.columns([1, 1, 5])
            if col1.button("✅ Ya, Hapus", use_container_width=True, type="primary"):
                with st.spinner(f"Menghapus '{file_name}' dan semua datanya..."):
                    success, message = delete_document_from_supabase(file_name)
                    st.toast(message, icon="✅" if success else "❌")
                    st.session_state.file_to_delete = None 
                    st.cache_data.clear()
                    st.rerun() 
            if col2.button("❌ Batal", use_container_width=True):
                st.session_state.file_to_delete = None
                st.rerun()

    # --- Daftar Dokumen ---
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
                                st.cache_data.clear()
                                st.rerun()
                        except Exception as e:
                            st.error(f"Gagal menambahkan pengguna: {e}")
                else:
                    st.warning("Username dan password tidak boleh kosong.")
    with col_user_2:
        st.subheader("Daftar Pengguna Saat Ini")
        users_data, _ = get_dashboard_data(supabase)
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
            st.session_state.chat_history.append(HumanMessage(content=user_question))
            with st.chat_message("user"):
                st.markdown(user_question)
            with st.spinner("DIGICHATBOT (Admin) sedang memproses..."):
                response = st.session_state.conversation_chain({'question': user_question})
                ai_answer = response.get('answer', 'Error')
                st.session_state.chat_history.append(AIMessage(content=ai_answer))
                st.rerun()