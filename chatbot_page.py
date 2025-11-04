# chatbot_page.py
import streamlit as st
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
import google.generativeai as genai
from langchain_community.vectorstores import SupabaseVectorStore
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_core.messages import AIMessage, HumanMessage
from langchain.prompts import PromptTemplate # <-- 1. IMPORT PROMPT TEMPLATE

# --- Fungsi Helper (Khusus Chatbot) ---

# --- PERUBAHAN BESAR DI FUNGSI INI ---
@st.cache_resource
def get_conversation_chain(_vectorstore, google_api_key):
    """
    Membuat chain percakapan yang menggunakan memori dan LLM
    dengan PROMPT KUSTOM.
    """
    genai.configure(api_key=google_api_key)
    llm = ChatGoogleGenerativeAI(
        model="models/gemini-2.5-pro",
        temperature=0.1, # Turunkan suhu agar lebih patuh
        convert_system_message_to_human=True,
        google_api_key=google_api_key
    )
    
    memory = ConversationBufferMemory(
        memory_key='chat_history', 
        return_messages=True, 
        output_key='answer' 
    )
    
    # 2. BUAT TEMPLATE PROMPT ANDA
    # Ini adalah "brief" yang Anda berikan, diubah menjadi instruksi
    CUSTOM_PROMPT_TEMPLATE = """
    Anda adalah asisten AI akademik yang sopan dan profesional. Jawab pertanyaan pengguna secara langsung dan to the point dan tetap sopan dan ramah, HANYA berdasarkan konteks yang diberikan di bawah ini.
    JANGAN pernah memulai jawaban Anda dengan frasa seperti "Berdasarkan konteks yang diberikan...".

    Konteks:
    {context}

    Pertanyaan:
    {question}

    Aturan Jawaban:
    1, jika jika ada jawaban di dalam konteks yang mendekati pertanyaan berikan saja jawaban yang ada di konteks dengan bilang "pada dokumen ini hanya di sebutkan..." jadi saya tidak menemukan informasi ....
    2. Jika jawaban ditemukan di dalam konteks, berikan jawaban tersebut secara jelas dan sopan serta ramah.
    3. Setelah memberikan jawaban (jika ditemukan), SELALU tambahkan kalimat di baris baru: "Apakah ada yang bisa saya bantu lagi?"
    4. Jika informasi tidak ada di dalam konteks atau Anda tidak tahu, jawab bahwa anda tidak tahu konteks tersebut dan bilang bahwa anda bisa tanyakan langsung ke Bu Intan."
    5. Jangan mengarang jawaban di luar konteks.

    Jawaban (langsung, sopan, dan ikuti aturan):
    """
    
    # 3. BUAT OBJEK PROMPT
    CUSTOM_PROMPT = PromptTemplate(
        template=CUSTOM_PROMPT_TEMPLATE, input_variables=["context", "question"]
    )

    # 4. SUNTIKKAN PROMPT KE DALAM CHAIN
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=_vectorstore.as_retriever(), 
        memory=memory,
        return_source_documents=True,
        combine_docs_chain_kwargs={"prompt": CUSTOM_PROMPT} # <-- Ini menyuntikkan prompt
    )
    return conversation_chain

# --- (Tidak ada perubahan di fungsi ini) ---
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

# --- (Tidak ada perubahan di fungsi ini) ---
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
    st.markdown("<h1 style='text_align: center;'>DIGICHATBOT</h1>", unsafe_allow_html=True)
    st.markdown("<div style='text_align: center;'>Tanyakan informasi akademik di sini</div>", unsafe_allow_html=True)
    st.write("---")

    # Tampilkan History DAN Sumber
    for message in st.session_state.chat_history:
        if isinstance(message, HumanMessage):
            with st.chat_message("user"): 
                st.markdown(message.content)
        elif isinstance(message, AIMessage):
            with st.chat_message("assistant"): 
                st.markdown(message.content)
                if "sources" in message.metadata and message.metadata["sources"]:
                    with st.expander("Lihat Sumber Dokumen"):
                        for source_file, public_url in message.metadata["sources"].items():
                            st.markdown(f"📄 [{source_file}]({public_url})")

    # Logika Input Baru
    user_question = st.chat_input("Ajukan pertanyaan disini...")

    if user_question:
        if st.session_state.conversation_chain is None:
            st.error("Sesi chat tidak terinisialisasi. Coba muat ulang.")
        else:
            st.session_state.chat_history.append(HumanMessage(content=user_question))
            
            with st.spinner("DIGICHATBOT sedang memproses..."):
                
                chain = st.session_state.conversation_chain
                response = chain({'question': user_question})
                
                ai_answer = response.get('answer', 'Maaf, saya tidak menemukan jawaban.')
                
                ai_sources = {}
                if 'source_documents' in response:
                    sources = {doc.metadata['source'] for doc in response['source_documents']}
                    if sources:
                        for source_file in sources:
                            try:
                                supabase = st.session_state['supabase']
                                public_url = supabase.storage.from_('pdf_documents').get_public_url(source_file)
                                ai_sources[source_file] = public_url
                            except Exception as e:
                                st.warning(f"Tidak dapat membuat link untuk {source_file}: {e}")

                # Jika jawaban adalah fallback "Bu Intan", jangan tampilkan sumber
                if "Bu Intan" in ai_answer:
                    ai_sources = {} # Kosongkan sumber

                st.session_state.chat_history.append(
                    AIMessage(
                        content=ai_answer, 
                        metadata={"sources": ai_sources} 
                    )
                )
                
                st.rerun()