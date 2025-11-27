# chatbot_page.py
import streamlit as st
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
import google.generativeai as genai
from langchain_community.vectorstores import SupabaseVectorStore
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
import time 

# --- Impor untuk LCEL (Streaming) ---
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

# ---  Impor untuk Re-ranking flaskrank---
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_community.document_compressors import FlashrankRerank


# --- Fungsi Helper untuk chatbotnya ---
@st.cache_resource
def get_conversation_chain(_vectorstore, google_api_key):
    """
    Membuat chain percakapan RAG yang STATELESS, STREAMABLE,
    dan menggunakan RE-RANKING.
    """
    genai.configure(api_key=google_api_key)
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.1,
        convert_system_message_to_human=True,
        google_api_key=google_api_key
    )
    
    # 1. SETUP RETRIEVER
    base_retriever = _vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={'k': 20}
    )
    compressor = FlashrankRerank(top_n=10)
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=base_retriever
    )

    # 2. PROMPT UNTUK REPHRASE PERTANYAAN
    REPHRASE_PROMPT_TEMPLATE = """
    Given the following conversation and a follow up question, rephrase the 
    follow up question to be a standalone question, in its original language.

    Chat History:
    {chat_history}

    Follow Up Input: {input}
    Standalone question:"""
    
    rephrase_prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", REPHRASE_PROMPT_TEMPLATE),
    ])
    
    # 3. BUAT HISTORY-AWARE RETRIEVER
    history_aware_retriever = create_history_aware_retriever(
        llm=llm,
        retriever=compression_retriever,
        prompt=rephrase_prompt
    )

    # 4. PROMPT JAWABAN AKHIR
    SYSTEM_PROMPT = """
    Anda adalah asisten AI akademik yang sopan dan profesional. Jawab pertanyaan pengguna secara langsung dan tetap sopan dan ramah, HANYA berdasarkan konteks yang diberikan di bawah ini.
    JANGAN pernah memulai jawaban Anda dengan frasa seperti Berdasarkan konteks/dokumen yang diberikan... dan lainnya.

    Konteks:
    {context}

    Aturan Jawaban:
    1. jika jika ada jawaban di dalam konteks yang mendekati pertanyaan berikan saja jawaban yang ada di konteks dengan bilang "pada dokumen ini hanya di sebutkan..." jadi saya tidak menemukan informasi ....
    2. Jika jawaban ditemukan di dalam konteks, berikan jawaban tersebut secara jelas dan sopan serta ramah.
    3. Setelah memberikan jawaban (jika ditemukan), SELALU tambahkan kalimat di baris baru: "Apakah ada yang bisa saya bantu lagi?"
    4. Jika informasi tidak ada di dalam konteks atau Anda tidak tahu, jawab bahwa anda tidak tahu konteks tersebut dan bilang bahwa anda bisa tanyakan langsung ke Bu Intan."
    5. Jangan mengarang jawaban di luar konteks.
    6. jika memungkinkan menjawab dengan list buatkan dengan list, pokoknya gimana caranya jawaban dapat dibaca dengan user dengan mudah

    Jawaban (langsung, sopan, dan ikuti aturan):
    """
    
    # QA Prompt
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}") 
    ])

    # 5. BUAT CHAIN UNTUK MENJAWAB PERTANYAAN
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

    # 6. GABUNGKAN SEMUA MENJADI RAG CHAIN FINAL
    rag_chain = create_retrieval_chain(
        history_aware_retriever,
        question_answer_chain
    )
    
    return rag_chain

def init_user_chat_session():
    """Inisialisasi session state yang diperlukan untuk halaman chat."""
    
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    # trigger pertanyaan cepat (FAQ)
    if 'prompt_trigger' not in st.session_state:
        st.session_state.prompt_trigger = None
    
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

# --- Fungsi helper untuk log ---
def save_chat_log(username, question, answer, response_time):
    """Fungsi untuk mengirim data log ke Supabase"""
    try:
        if 'supabase' in st.session_state:
            data = {
                "username": username,
                "question": question,
                "answer": answer,
                "response_time": response_time
            }
            st.session_state['supabase'].table('chat_logs').insert(data).execute()
    except Exception as e:
        print(f"[Error Log] Gagal menyimpan log: {e}")

def show_chatbot_page():
    """Menampilkan halaman chatbot untuk pengguna biasa."""
    
    init_user_chat_session()
    
    # --- Sidebar ---
    with st.sidebar:
        st.markdown(f"### Selamat Datang, {st.session_state['username']}!")
        st.write("---")
        st.markdown("<div style='flex-grow: 1;'></div>", unsafe_allow_html=True)
        if st.button("Log Out", use_container_width=True):
            st.session_state.logged_in = False
            st.session_state.username = None
            st.session_state.chat_history = []
            st.session_state.conversation_chain = None
            st.session_state.prompt_trigger = None
            st.rerun()
    
    # --- Area Chat Header ---
    st.markdown("<h1 style='text-align: center;'>DIGICHATBOT</h1>", unsafe_allow_html=True)
    st.markdown("<div style='text-align: center;'>Tanyakan informasi akademik di sini</div>", unsafe_allow_html=True)
    st.write("---")

    # --- FAQ Buttons ---
    if not st.session_state.chat_history:
        st.caption("Pertanyaan yang sering ditanyakan (Klik untuk kirim):")
        col_faq1, col_faq2 = st.columns(2)
        
        faq_questions = [
            "Gimana cara konversi magang?",
            "Berikan saya link perbaikan absen.",
            "Gimana cara daftar sidang akhir?",
            "Bagaimana cara daftar SUP?"
        ]
        
        for i, question in enumerate(faq_questions):
            # Mengatur kolom selang-seling
            with col_faq1 if i % 2 == 0 else col_faq2:
                if st.button(question, use_container_width=True):
                    # Set trigger dan reload halaman agar diproses seperti input user
                    st.session_state.prompt_trigger = question
                    st.rerun()
        st.write("")

    # --- Tampilkan Riwayat Chat ---
    for message in st.session_state.chat_history:
        if isinstance(message, HumanMessage):
            with st.chat_message("user"): 
                st.markdown(message.content)
        elif isinstance(message, AIMessage):
            with st.chat_message("assistant"): 
                st.markdown(message.content)
                if "response_time" in message.metadata and message.metadata["response_time"] is not None:
                    st.markdown(f"<p style='text-align: right; font-size: 0.75em; color: #888888; margin-top: 5px; opacity: 0.7;'>Terjawab dalam {message.metadata['response_time']:.2f} detik</p>", unsafe_allow_html=True)
                if "sources" in message.metadata and message.metadata["sources"]:
                    with st.expander("Lihat Sumber Dokumen"):
                        for source_file, public_url in message.metadata["sources"].items():
                            st.markdown(f"📄 [{source_file}]({public_url})")

    # --- Logika Input Gabungan dari Ketik Manual dan Tombol FAQ ---
    
    # 1. Cek apakah ada trigger dari tombol FAQ
    triggered_question = st.session_state.get("prompt_trigger")
    
    # Reset trigger agar tidak looping
    if triggered_question:
        st.session_state.prompt_trigger = None 
        
    # 2. Cek input manual dari user
    chat_input_value = st.chat_input("Ajukan pertanyaan disini...")

    # 3. Tentukan pertanyaan final (Prioritas: Trigger > Input Manual)
    user_question = triggered_question if triggered_question else chat_input_value

    if user_question:
        if st.session_state.conversation_chain is None:
            st.error("Sesi chat tidak terinisialisasi. Coba muat ulang.")
        else:
            # 1. Tambahkan pesan pengguna ke history
            st.session_state.chat_history.append(HumanMessage(content=user_question))
            with st.chat_message("user"):
                st.markdown(user_question)
            
            # 2. Streaming respons AI
            with st.chat_message("assistant"):
                placeholder = st.empty()
                full_response = ""
                source_documents = []
                response_time = 0.0
                
                start_time = time.time()

                try:
                    chat_history_for_chain = st.session_state.chat_history[:-1]
                    
                    stream = st.session_state.conversation_chain.stream({
                        "input": user_question,
                        "chat_history": chat_history_for_chain
                    })

                    for chunk in stream:
                        if "answer" in chunk and chunk["answer"]:
                            full_response += chunk["answer"]
                            placeholder.markdown(full_response + "▌")
                        if "context" in chunk:
                            source_documents = chunk["context"]
                    
                    end_time = time.time()
                    response_time = end_time - start_time
                    
                    placeholder.markdown(full_response)
                    
                    # Tampilkan waktu
                    st.markdown(f"<p style='text-align: right; font-size: 0.75em; color: #888888; margin-top: 5px; opacity: 0.7;'>Terjawab dalam {response_time:.2f} detik</p>", unsafe_allow_html=True)

                except Exception as e:
                    st.error(f"Terjadi kesalahan: {e}")
                    full_response = "Maaf, saya mengalami gangguan. Silakan coba lagi atau tanyakan Bu Intan."
                    placeholder.markdown(full_response)
                    response_time = 0.0

                # --- MENYIMPAN LOG KE SUPABASE ---
                save_chat_log(
                    username=st.session_state.get('username', 'Anonymous'),
                    question=user_question,
                    answer=full_response,
                    response_time=response_time
                )

                # 3. Proses Sumber
                ai_sources = {}
                if "Bu Intan" not in full_response and source_documents:
                    sources = {doc.metadata['source'] for doc in source_documents}
                    if sources:
                        try:
                            supabase = st.session_state['supabase']
                            for source_file in sources:
                                public_url = supabase.storage.from_('pdf_documents').get_public_url(source_file)
                                ai_sources[source_file] = public_url
                        except Exception as e:
                            pass 
                    
                    if ai_sources:
                        with st.expander("Lihat Sumber Dokumen"):
                            for source_file, public_url in ai_sources.items():
                                st.markdown(f"📄 [{source_file}]({public_url})")

                # 4. Append history
                st.session_state.chat_history.append(
                    AIMessage(
                        content=full_response, 
                        metadata={"sources": ai_sources, "response_time": response_time} 
                    )
                )