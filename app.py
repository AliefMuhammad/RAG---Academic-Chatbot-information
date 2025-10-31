import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
import google.generativeai as genai
from langchain_community.vectorstores import SupabaseVectorStore
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from dotenv import load_dotenv
from supabase.client import Client, create_client
from langchain.docstore.document import Document
from langchain_core.messages import AIMessage, HumanMessage

# --- Konfigurasi Awal ---
load_dotenv()

# Konfigurasi Google API (Digunakan untuk Embeddings dan LLM)
google_api_key = os.getenv("GOOGLE_API_KEY")
if not google_api_key:
    st.error("GOOGLE_API_KEY tidak ditemukan")
    st.stop()
genai.configure(api_key=google_api_key)

# Konfigurasi Supabase (Tidak berubah)
supabase_url = os.getenv("SUPABASE_URL")
supabase_key = os.getenv("SUPABASE_ANON_KEY")
if not supabase_url or not supabase_key:
    st.error("SUPABASE_URL atau SUPABASE_ANON_KEY tidak ditemukan")
    st.stop()

# Inisialisasi Supabase Client
supabase: Client = create_client(supabase_url, supabase_key)

# --- Fungsi-Fungsi Inti (Tidak Berubah) ---

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
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
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

# --- PERUBAHAN UTAMA: Menggunakan Google Gemini Langsung untuk LLM ---
@st.cache_resource
def get_conversation_chain(_vectorstore):
    """
    Membuat chain percakapan yang menggunakan memori dan LLM langsung dari Google Gemini.
    """
    llm = ChatGoogleGenerativeAI(
        model="models/gemini-2.5-pro",
        temperature=0.3,
        convert_system_message_to_human=True
    )
    
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=_vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain

def get_uploaded_filenames():
    """Mengambil nama file unik yang sudah ada di Supabase."""
    try:
        response = supabase.table('documents').select('metadata', count='exact').execute()
        filenames = {item['metadata']['source'] for item in response.data if 'source' in item['metadata']}
        return sorted(list(filenames))
    except Exception as e:
        st.error(f"Gagal mengambil daftar file dari Supabase: {e}")
        return []

def delete_document_from_supabase(filename):
    """Menghapus semua entri yang terkait dengan nama file tertentu dari Supabase."""
    try:
        supabase.table('documents').delete().eq('metadata->>source', filename).execute()
        return True, f"Dokumen '{filename}' dan datanya berhasil dihapus."
    except Exception as e:
        return False, f"Gagal menghapus dokumen dari Supabase: {e}"

# --- Tampilan Streamlit (UI) ---

def main():
    st.set_page_config(page_title="Chatbot Akademik FEB", layout="wide")
    st.markdown("<h1 style='text-align: center;'>DIGICHATBOT</h1>", unsafe_allow_html=True)
    st.markdown("<div style='text-align: center;'>Tanyakan informasi akademik di sini</div>", unsafe_allow_html=True)
    st.write("---")

    # Inisialisasi session state
    if 'file_to_delete' not in st.session_state:
        st.session_state.file_to_delete = None
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'conversation_chain' not in st.session_state:
        st.session_state.conversation_chain = None

    # Membuat vector store sekali saja
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = SupabaseVectorStore(
        client=supabase,
        embedding=embeddings,
        table_name="documents",
        query_name="match_documents"
    )
    st.session_state.conversation_chain = get_conversation_chain(vector_store)

    # Dialog Konfirmasi Hapus
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

    # Tampilan Antarmuka Chat
    for message in st.session_state.chat_history:
        if isinstance(message, HumanMessage):
            with st.chat_message("user", avatar="🧑‍💻"):
                st.markdown(message.content)
        elif isinstance(message, AIMessage):
            with st.chat_message("assistant", avatar="🤖"):
                st.markdown(message.content)

    # Input dari pengguna
    user_question = st.chat_input("Ajukan pertanyaan disini...")

    if user_question:
        with st.spinner("Memproses..."):
            response = st.session_state.conversation_chain({'question': user_question})
            st.session_state.chat_history = response['chat_history']
            st.rerun()

    # Sidebar
    with st.sidebar:
        st.title("Menu")
        st.markdown("---")
        
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

if __name__ == "__main__":
    main()