# admin_page.py
import streamlit as st
import tempfile
import os
import logging
import traceback 
import hashlib
import uuid

# --- Import unstructured ---
from unstructured.partition.pdf import partition_pdf
from unstructured.cleaners.core import clean_extra_whitespace, clean_non_ascii_chars

# --- Import Library Lain ---
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import SupabaseVectorStore
from langchain_core.documents import Document

# --- Setup Logger ---
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

logging.getLogger("pdfminer").setLevel(logging.ERROR)

# --- Fungsi Akuisisi Data dengan LOGGING (TETAP) ---

def process_and_store_document(pdf_file_object, file_content, file_name, classification, file_hash):
    """
    Fungsi master untuk memproses satu PDF dengan logging langkah-demi-langkah.
    """
    supabase = st.session_state['supabase']
    google_api_key = st.session_state['google_api_key']
    parent_id = str(uuid.uuid4())
    
    logger.info(f"[START] Memulai proses untuk file: {file_name} (ID: {parent_id})")

    try:
        # --- Langkah 1: Parsing PDF dengan Unstructured ---
        st.write(f"Langkah 1/4: Memulai 'unstructured' parsing untuk {file_name}...")
        logger.info(f"[Langkah 1] Memulai partition_pdf (strategy=hi_res) untuk {file_name}")
        
        elements = partition_pdf(
            file=pdf_file_object,
            strategy="hi_res", 
            infer_table_structure=True,
            languages=["ind", "eng"], 
            extract_images_in_pdf=False,
            
            # --- new chunking metodh (smart chunk)---
            chunking_strategy="by_title",
            max_characters=4000,
            combine_text_under_n_chars=2000,
            new_after_n_chars=3800,
        )
        
        st.write(f"Parsing selesai. Ditemukan {len(elements)} CHUNKS logis.")
        logger.info(f"partition_pdf selesai. Menemukan {len(elements)} elemen.")

        # --- Langkah 2: Buat Dokumen LangChain dari Elemen/Chunks ---
        st.write("Langkah 2/4: Membersihkan teks dan membuat Dokumen LangChain...")
        logger.info(f"[Langkah 2] Memulai pembersihan dan pembuatan Dokumen LangChain...")
        documents = []
        for el in elements:
            element_type = el.category
            page_number = el.metadata.page_number
            
            if element_type == "Table":
                text = el.metadata.text_as_html
            else:
                text = str(el)

            text = clean_non_ascii_chars(text)
            text = clean_extra_whitespace(text)
            
            if not text.strip():
                continue 

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
            logger.warning(f"Tidak ada teks yang diekstrak dari {file_name}. Proses dihentikan.")
            return False, f"Tidak ada teks yang bisa diekstrak dari {file_name}."

        st.write(f"Selesai membuat {len(documents)} dokumen. Memulai penyimpanan...")
        logger.info(f"Selesai membuat {len(documents)} dokumen.")

        # --- Langkah 3: Simpan ke Database (Atomik) ---
        
        # 3a. Buat entri di tabel file induk (parent_files)
        st.write("Langkah 3/4: Menyimpan data ke Database...")
        logger.info(f"[Langkah 3a] Menyimpan record ke tabel 'parent_files'...")
        supabase.table('parent_files').insert({
            "id": parent_id,
            "file_name": file_name,
            "file_hash": file_hash,
            "classification": classification
        }).execute()
        logger.info("Sukses menyimpan ke 'parent_files'.")
        
        # 3b. Unggah file PDF asli ke Supabase Storage
        logger.info(f"[Langkah 3b] Mengunggah file '{file_name}' ke Supabase Storage...")
        try:
            supabase.storage.from_('pdf_documents').upload(
                path=file_name,
                file=file_content,
                file_options={"content-type": "application/pdf", "upsert": "true"}
            )
            logger.info("Sukses mengunggah ke Storage.")
        except Exception as storage_error:
            if "Duplicate" not in str(storage_error) and "409" not in str(storage_error):
                raise storage_error 
            logger.warning(f"File {file_name} sudah ada di Storage. Melanjutkan...")

        # 3c. Simpan text chunks dan embeddings ke Database (tabel documents)
        st.write("Langkah 4/4: Membuat embedding dan menyimpannya ke Vector Store...")
        logger.info(f"[Langkah 3c] Mengkonfigurasi GoogleGenerativeAIEmbeddings...")
        genai.configure(api_key=google_api_key)
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=google_api_key)
        
        logger.info("Memulai SupabaseVectorStore.from_documents... (Ini adalah proses embedding & insert ke DB)")
        SupabaseVectorStore.from_documents(
            documents=documents,
            embedding=embeddings,
            client=supabase,
            table_name="documents",
            query_name="match_documents",
            chunk_size=500,
        )
        logger.info(f"Sukses! Embedding dan penyimpanan vektor selesai untuk {file_name}.")
        
        st.success(f"File '{file_name}' berhasil diproses (RAG dan Storage)!")
        logger.info(f"[END] Pemrosesan {file_name} (ID: {parent_id}) - SUKSES")
        return True, f"Sukses memproses {file_name}"

    except Exception as e:
        # --- Logging Eror 
        st.error(f"GAGAL TOTAL memproses '{file_name}'.")

        logger.error(f"GAGAL TOTAL memproses {file_name} (ID: {parent_id}). Eror: {str(e)}")
        logger.error(f"Traceback lengkap:\n{traceback.format_exc()}")

        if "importlib" in str(e):
             st.error(f"Eror Kritis (Module): {e}. Coba perbaiki environment Anda. Jalankan: pip install \"importlib-metadata<5.0.0\"")
        elif "poppler" in str(e):
             st.error(f"Eror Kritis (Dependensi): {e}. Pastikan 'poppler' terinstal. Jalankan: brew install poppler")
        else:
            with st.expander("Klik untuk melihat detail eror teknis"):
                st.code(traceback.format_exc())

        logger.info(f"Memulai rollback untuk {parent_id}...")
        try:
            supabase.table('parent_files').delete().eq('id', parent_id).execute()
            logger.info(f"Rollback {parent_id} berhasil.")
        except Exception as rollback_e:
            st.error(f"Gagal melakukan rollback. Silakan hapus manual parent_id: {parent_id}")
            logger.error(f"GAGAL ROLLBACK untuk {parent_id}. Eror: {rollback_e}")
            
        return False, str(e)


# --- connect to db Supabase ---

@st.cache_data(ttl=60)
def get_existing_file_hashes(_supabase):
    """[BARU] Mengambil semua hash file yang sudah ada dari tabel parent_files."""
    logger.info("Mengambil cache hash file...")
    try:
        response = _supabase.table('parent_files').select('file_hash').execute()
        if response.data:
            return set(item['file_hash'] for item in response.data)
        return set()
    except Exception as e:
        st.error(f"Gagal mengambil hash file: {e}")
        logger.error(f"Gagal mengambil hash file: {e}")
        return set()

@st.cache_data(ttl=60)
def get_document_list(_supabase):
    """[MODIFIKASI] Mengambil daftar dokumen dari tabel 'parent_files'."""
    logger.info("Mengambil daftar dokumen...")
    try:
        response = _supabase.table('parent_files').select('file_name, classification').order('created_at', desc=True).execute()
        if response.data:
            return [{"source": item['file_name'], "classification": item['classification']} for item in response.data]
        return []
    except Exception as e:
        st.error(f"Gagal mengambil daftar file dari Supabase: {e}")
        logger.error(f"Gagal mengambil daftar file: {e}")
        return []

def delete_document_from_supabase(filename):
    """[MODIFIKASI] Menghapus file dari DB (parent & chunks) DAN Storage."""
    supabase = st.session_state['supabase']
    logger.info(f"Memulai proses hapus untuk: {filename}")
    try:
        parent_response = supabase.table('parent_files').select('id').eq('file_name', filename).execute()
        if not parent_response.data:
            logger.warning(f"File '{filename}' tidak ditemukan di parent_files saat akan dihapus.")
            return False, f"File '{filename}' tidak ditemukan di tabel parent_files."
        
        parent_id = parent_response.data[0]['id']

        logger.info(f"Menghapus chunks untuk parent_id: {parent_id}...")
        supabase.table('documents').delete().eq('metadata->>parent_file_id', parent_id).execute()
        
        logger.info(f"Menghapus parent record untuk {filename}...")
        supabase.table('parent_files').delete().eq('id', parent_id).execute()
        
        logger.info(f"Menghapus file dari Storage...")
        supabase.storage.from_('pdf_documents').remove([filename])
        
        logger.info(f"Sukses menghapus {filename}")
        return True, f"Dokumen '{filename}' (dan semua chunks-nya) berhasil dihapus."
    except Exception as e:
        if "No such object" in str(e):
             logger.warning(f"Sukses hapus {filename} dari DB, tapi tidak ada di Storage.")
             return True, f"Dokumen '{filename}' berhasil dihapus dari DB (tidak ditemukan di Storage)."
        logger.error(f"Gagal menghapus {filename}: {e}\n{traceback.format_exc()}")
        return False, f"Gagal menghapus dokumen: {e}"


# --- Tampilan Utama Halaman Admin (CLEAN VERSION) ---
def show_admin_clean():
    """Menampilkan halaman admin hanya untuk manajemen dokumen (Upload & List/Hapus)."""
    
    # Inisialisasi variabel state untuk hapus file
    if 'file_to_delete' not in st.session_state:
        st.session_state.file_to_delete = None
        
    supabase = st.session_state['supabase']

    # --- Sidebar Navigasi Sederhana ---
    with st.sidebar:
        st.markdown(f"### Admin Panel, {st.session_state['username']}!")
        st.write("---")
        st.info("Mode Admin: Manajemen Dokumen")
        st.markdown("<div style='flex-grow: 1;'></div>", unsafe_allow_html=True)
        if st.button("Log Out", use_container_width=True):
            st.session_state.logged_in = False
            st.session_state.username = None
            st.session_state.file_to_delete = None
            st.rerun()

    # --- Judul Utama ---
    st.markdown("<h1 style='text-align: center;'>DIGICHATBOT (ADMIN)</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center;'>Panel Manajemen Dokumen Pengetahuan</p>", unsafe_allow_html=True)
    st.write("---")
    
    # --- 1. Bagian Upload Dokumen ---
    st.markdown('<h2>⬆️ Unggah Dokumen Baru</h2>', unsafe_allow_html=True)
    
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
                    logger.info(f"Mulai loop untuk file: {sanitized_name}")
                    st.markdown(f"--- \n ### Memproses: {sanitized_name}")
                    
                    pdf.seek(0)
                    file_content = pdf.read()
                    file_hash = hashlib.md5(file_content).hexdigest()
                    
                    if file_hash in existing_hashes:
                        st.warning(f"File '{sanitized_name}' sudah ada di database. Proses dilewati.")
                        continue
                        
                    pdf.seek(0)
                    
                    # --- Memanggil fungsi processing ---
                    success, message = process_and_store_document(
                        pdf, 
                        file_content, 
                        sanitized_name, 
                        classification, 
                        file_hash
                    )
                    
                    if not success:
                        all_successful = False
                        st.error(f"Gagal memproses {sanitized_name}. Lihat detail di atas.")
                    else:
                        existing_hashes.add(file_hash)

            if all_successful:
                st.success("Semua file baru berhasil diproses!")
                st.cache_data.clear()
                st.rerun()
            else:
                st.error("Satu atau lebih file gagal diproses.")
                st.cache_data.clear()
                st.rerun()
    
    st.divider()

    # --- 2. Bagian List & Hapus Dokumen ---
    st.markdown('<h2>📂 Database Dokumen Saat Ini</h2>', unsafe_allow_html=True)

    # --- Bagian Konfirmasi Hapus (Modal-like) ---
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

    # --- Render Daftar Dokumen ---
    document_list = get_document_list(supabase) 
    if document_list:
        # Menggunakan container dengan scroll jika dokumen banyak
        with st.container(height=500): 
            for meta in document_list:
                doc_name = meta.get('source', 'Nama Tidak Ditemukan')
                doc_class = meta.get('classification', 'Belum Terklasifikasi')
                
                try:
                    file_url = supabase.storage.from_('pdf_documents').get_public_url(doc_name)
                except Exception:
                    file_url = "#"
                
                col1, col2 = st.columns([4, 1])
                with col1:
                    st.info(f"📄 [**{doc_name}**]({file_url})\n\n*Klasifikasi: {doc_class}*")
                    
                with col2:
                    is_modal_active = st.session_state.file_to_delete is not None
                    # Tombol Hapus
                    if st.button("🗑️", key=f"delete_{doc_name}", help=f"Hapus {doc_name}", use_container_width=True, disabled=is_modal_active):
                        st.session_state.file_to_delete = doc_name
                        st.rerun()
    else:
        st.info("Belum ada dokumen di database.")