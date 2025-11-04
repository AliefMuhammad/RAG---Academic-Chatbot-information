# chatbot_page.py
import streamlit as st
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
import google.generativeai as genai
from langchain_community.vectorstores import SupabaseVectorStore
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_core.messages import AIMessage, HumanMessage

# --- Fungsi Helper (Khusus Chatbot) ---

@st.cache_resource
def get_conversation_chain(_vectorstore, google_api_key):
    """
    Membuat chain percakapan yang menggunakan memori dan LLM.
    """
    genai.configure(api_key=google_api_key)
    llm = ChatGoogleGenerativeAI(
        model="models/gemini-2.5-pro",
        temperature=0.3,
        convert_system_message_to_human=True,
        google_api_key=google_api_key
    )
    
    # PERBAIKAN: Menambahkan output_key='answer'
    memory = ConversationBufferMemory(
        memory_key='chat_history', 
        return_messages=True, 
        output_key='answer' 
    )
    
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=_vectorstore.as_retriever(), 
        memory=memory,
        return_source_documents=True # PERBAIKAN: Meminta sumber dokumen
    )
    return conversation_chain

def init_user_chat_session():
    """Inisialisasi session state yang diperlukan untuk halaman chat."""
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    if 'conversation_chain' not in st.session_state or st.session_state.conversation_chain is None:
        
        if 'supabase' not in st.session_state or 'google_api_key' not in st.session_state:
            st.error("Sesi tidak valid. Silakan Log Out dan Login kembali.")
            st.stop()
            
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

# --- Tampilan Utama Halaman Chatbot ---

def show_chatbot_page():
    """Menampilkan halaman chatbot untuk pengguna biasa."""
    
    init_user_chat_session()
    
    # --- Sidebar Minimalis ---
    with st.sidebar:
        st.markdown(f"### Selamat Datang, {st.session_state['username']}!")
        st.write("---")
        st.markdown("<div style='flex-grow: 1;'></div>", unsafe_allow_html=True)
        
        if st.button("Log Out", use_container_width=True):
            st.session_state.logged_in = False
            st.session_state.username = None
            st.session_state.chat_history = []
            st.session_state.conversation_chain = None
            st.rerun()
    
    # --- Area Chat Utama ---
    st.markdown("<h1 style='text-align: center;'>DIGICHATBOT</h1>", unsafe_allow_html=True)
    st.markdown("<div style='text-align: center;'>Tanyakan informasi akademik di sini</div>", unsafe_allow_html=True)
    st.write("---")

    # --- PERBAIKAN: Tampilkan History DAN Sumber ---
    for message in st.session_state.chat_history:
        if isinstance(message, HumanMessage):
            with st.chat_message("user"): 
                st.markdown(message.content)
        elif isinstance(message, AIMessage):
            with st.chat_message("assistant"): 
                st.markdown(message.content)
                # Cek jika ada metadata sumber di pesan AI
                if "sources" in message.metadata and message.metadata["sources"]:
                    with st.expander("Lihat Sumber Dokumen"):
                        for source_file, public_url in message.metadata["sources"].items():
                            st.markdown(f"📄 [{source_file}]({public_url})")

    # --- PERBAIKAN: Logika Input Baru ---
    user_question = st.chat_input("Ajukan pertanyaan disini...")

    if user_question:
        if st.session_state.conversation_chain is None:
            st.error("Sesi chat tidak terinisialisasi. Coba muat ulang.")
        else:
            # 1. Tambahkan pertanyaan user ke history (untuk ditampilkan di rerun berikutnya)
            st.session_state.chat_history.append(HumanMessage(content=user_question))
            
            # 2. Tampilkan spinner kustom
            with st.spinner("DIGICHATBOT sedang memproses..."):
                
                # 3. Panggil RAG Chain
                chain = st.session_state.conversation_chain
                response = chain({'question': user_question})
                
                # 4. Ekstrak jawaban
                ai_answer = response.get('answer', 'Maaf, saya tidak menemukan jawaban.')
                
                # 5. Ekstrak dan proses sumber
                ai_sources = {}
                if 'source_documents' in response:
                    # Buat set unik dari nama file sumber
                    sources = {doc.metadata['source'] for doc in response['source_documents']}
                    if sources:
                        for source_file in sources:
                            try:
                                supabase = st.session_state['supabase']
                                # PERBAIKAN: Menggunakan .from_() untuk supabase-py v2
                                public_url = supabase.storage.from_('pdf_documents').get_public_url(source_file)
                                ai_sources[source_file] = public_url
                            except Exception as e:
                                st.warning(f"Tidak dapat membuat link untuk {source_file}: {e}")

                # 6. Tambahkan jawaban AI (lengkap dengan metadata sumber) ke history
                st.session_state.chat_history.append(
                    AIMessage(
                        content=ai_answer, 
                        metadata={"sources": ai_sources} # Simpan sumber di metadata
                    )
                )
                
                # 7. Rerun halaman.
                st.rerun()