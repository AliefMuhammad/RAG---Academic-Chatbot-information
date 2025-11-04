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
    
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=_vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain

def init_user_chat_session():
    """Inisialisasi session state yang diperlukan untuk halaman chat."""
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'conversation_chain' not in st.session_state:
        # Membuat vector store
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
    st.set_page_config(page_title="Chatbot Akademik", layout="wide")
    
    # Panggil inisialisasi
    init_user_chat_session()
    
    # --- Tombol Logout ---
    with st.container():
        col1, col2 = st.columns([10, 1])
        with col1:
            st.markdown(f"<h3 style='margin-bottom: 0;'>Selamat Datang, {st.session_state['username']}!</h3>", unsafe_allow_html=True)
        with col2:
            if st.button("Log Out", use_container_width=True):
                st.session_state.logged_in = False
                st.session_state.username = None
                st.session_state.chat_history = []
                st.session_state.conversation_chain = None
                st.rerun()
    
    st.markdown("<h1 style='text-align: center;'>DIGICHATBOT</h1>", unsafe_allow_html=True)
    st.markdown("<div style='text-align: center;'>Tanyakan informasi akademik di sini</div>", unsafe_allow_html=True)
    st.write("---")

    # --- Tampilan ui Chat ---
    for message in st.session_state.chat_history:
        if isinstance(message, HumanMessage):
            with st.chat_message("user", avatar="🧑‍💻"):
                st.markdown(message.content)
        elif isinstance(message, AIMessage):
            with st.chat_message("assistant", avatar="🤖"):
                st.markdown(message.content)

    # --- Input Pengguna ---
    user_question = st.chat_input("Ajukan pertanyaan disini...")

    if user_question:
        with st.spinner("Memproses..."):
            response = st.session_state.conversation_chain({'question': user_question})
            st.session_state.chat_history = response['chat_history']
            st.rerun()