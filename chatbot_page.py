# chatbot_page.py
import streamlit as st
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
import google.generativeai as genai
from langchain_community.vectorstores import SupabaseVectorStore
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
import time 
import os

# --- Import langchain ---
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

# ---  Import Re-ranking flaskrank---
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_community.document_compressors import FlashrankRerank
from flashrank import Ranker
from langchain_core.callbacks import BaseCallbackHandler

# --- CLASS DEBUG LOGGING ---
class PrintRetrievalHandler(BaseCallbackHandler):
    """
    Callback custom untuk print log dokumen yang di-retrieve ke Terminal VS Code.
    Akan berjalan setiap kali retriever selesai mengambil dokumen.
    """
    def __init__(self):
        self.step_count = 0

    def on_retriever_end(self, documents, **kwargs):
        self.step_count += 1
        print(f"\n\n{('='*20)} [DEBUG RETRIEVER STEP {self.step_count}] {('='*20)}")
        print(f"📊 Total Chunk Ditemukan: {len(documents)}")
        
        print(f"\n🔍 TOP 5 CHUNKS (Setelah Rerank):")
        for i, doc in enumerate(documents[:10]): # Hanya ambil top 5 untuk di print
            source = doc.metadata.get('source', 'Unknown File')
            page = doc.metadata.get('page', '?')
            # Bersihkan newline agar terminal rapi
            preview = doc.page_content.replace('\n', ' ')[:200] 
            
            print(f"[{i+1}] 📄 {source} (Hal: {page})")
            print(f"    📝 Isi: {preview}...")
            print(f"    {'-'*40}")
        print(f"{('='*60)}\n")

# 1. HELPER FUNCTIONS
def get_all_available_documents(supabase_client):
    """
    Mengambil daftar semua nama file dan klasifikasinya dari database.
    Ini berfungsi sebagai 'Daftar Isi' untuk chatbot.
    """
    try:
        # Ambil file_name dan classification dari tabel parent_files
        response = supabase_client.table('parent_files').select('file_name, classification').execute()
        
        if not response.data:
            return "Belum ada dokumen yang tersedia."
            
        # Format menjadi string list yang rapi
        doc_list = []
        for item in response.data:
            clean_name = item['file_name'].replace('.pdf', '').replace('_', ' ')
            doc_list.append(f"- {clean_name} (Kategori: {item['classification']})")
            
        return "\n".join(doc_list)
    except Exception as e:
        return f"Gagal memuat daftar dokumen: {e}"

@st.cache_resource
def get_conversation_chain(_vectorstore, google_api_key, _supabase_client): 
    
    # 1. Ambil list Dokumen untuk Konteks Global
    available_docs_text = get_all_available_documents(_supabase_client)
    
    genai.configure(api_key=google_api_key)
    
    # Gunakan temperature sedikit lebih tinggi (0.3) agar kreatif saat rephrasing query
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.3,
        convert_system_message_to_human=True,
        google_api_key=google_api_key
    )
    
    # 2. OPTIMALISASI RETRIEVER (PERBAIKAN UTAMA)
    # Masalah sebelumnya: Dokumen relevan (beasiswa) tertimbun dokumen admin di ranking awal.
    # Solusi: Perbesar fetch_k agar pencarian awal Supabase lebih luas sebelum difilter MMR.
    base_retriever = _vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            'k': 50,           # Kirim 50 kandidat ke Reranker (Flashrank)
            'fetch_k': 100,    # Cari 300 dokumen teratas di DB (agar yang relevan tidak hilang)
            'lambda_mult': 0.7 # Balance antara relevansi dan keberagaman
        }
    )

    # Setup Flashrank
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_cache_path = os.path.join(current_dir, "model_cache")
    if not os.path.exists(model_cache_path):
        os.makedirs(model_cache_path)

    manual_ranker = Ranker(model_name="ms-marco-MultiBERT-L-12", cache_dir=model_cache_path)
    
    # Ambil top 20 hasil terbaik setelah reranking untuk dikirim ke LLM
    compressor = FlashrankRerank(client=manual_ranker, top_n=20)
    
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=base_retriever
    )

    # 3. REPHRASE PROMPT (QUERY EXPANSION)
    # Prompt ini memaksa LLM melihat daftar dokumen yang tersedia.
    
    REPHRASE_PROMPT_TEMPLATE = """
    You are an intelligent query optimizer for an academic retrieval system.
    
    CONTEXT (AVAILABLE DOCUMENTS IN DATABASE):
    {doc_list}

    TASK:
    Given the chat history and the latest user input, rephrase the follow up question 
    to be a STANDALONE question that is optimized for vector retrieval.
    
    RULES:
    1. If the user asks a BROAD question, you MUST explicitly include the NAMES of the relevant documents found in the 'CONTEXT' list above into the standalone question.
    2. Keep the language Indonesian.

    Chat History: {chat_history}
    Follow Up Input: {input}
    Standalone question:"""
    
    # Inject doc_list menggunakan .partial()
    rephrase_prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", REPHRASE_PROMPT_TEMPLATE),
    ]).partial(doc_list=available_docs_text)
    
    history_aware_retriever = create_history_aware_retriever(
        llm=llm,
        retriever=compression_retriever,
        prompt=rephrase_prompt
    )

    # 4. SYSTEM PROMPT (QA CHAIN)
    # Hapus 'f' string untuk menghindari konflik variabel {context}
    SYSTEM_PROMPT_TEMPLATE = """
    Anda adalah "DigiChatbot", asisten AI akademik Prodi Bisnis Digital Universitas Padjadjaran.
    
    INFORMASI GLOBAL (DAFTAR DOKUMEN YANG TERSEDIA DI DATABASE):
    Berikut adalah daftar file yang dimiliki sistem saat ini. Gunakan daftar ini untuk mengetahui konteks apa saja yang tersedia, meskipun detail isinya belum tentu muncul di hasil pencarian.
    {doc_list}
    
    KONTEKS PENCARIAN (DETAIL ISI DOKUMEN DARI DATABASE):
    {context}

    ATURAN MENJAWAB:
    1. Jangan menjawab "berdasarkan informasi yang diberikan" tapi menjawablah dengan natural seolah-olah anda mengetahuinya.
    2. PRIORITAS: Jika user bertanya "Ada apa saja?" atau "List dokumen", WAJIB melihat bagian "INFORMASI GLOBAL" di atas.
    3. Jika detail lengkap (seperti syarat/tanggal) dari salah satu dokumen tidak ditemukan di bagian "KONTEKS PENCARIAN", katakan jujur: "Saya melihat ada dokumen [Nama Dokumen] di sistem, namun detail isinya tidak terambil saat ini. Coba tanyakan lagi spesifik tentang [Nama Dokumen] tersebut."
    4. Jangan mengarang syarat/tanggal jika tidak ada di teks.
    5. Jawab dengan LIST (Bullet points) agar mudah dibaca.
    
    Jawaban:
    """
    
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT_TEMPLATE),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}") 
    ])
    
    # Inject doc_list ke system prompt juga
    qa_prompt = qa_prompt.partial(doc_list=available_docs_text)

    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

    rag_chain = create_retrieval_chain(
        history_aware_retriever,
        question_answer_chain
    )
    
    return rag_chain


def init_user_chat_session():
    """Inisialisasi session state."""
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
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
        st.session_state.conversation_chain = get_conversation_chain(vector_store, google_api_key, supabase)

def save_chat_log(username, question, answer, response_time):
    """Mengirim log ke Supabase."""
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

# 2. UI (HTML FORMATTER)

def format_ai_message(content, response_time=None, sources=None):
    # 1. HTML Waktu
    time_html = ""
    if response_time:
        time_html = f"""<div class="ai-time">⏱ {response_time:.2f}s</div>"""

    # 2. HTML Sumber
    sources_html = ""
    if sources:
        links = ""
        for filename, url in sources.items():
            links += f'<a href="{url}" target="_blank">📄 {filename}</a>'
        
        sources_html = f"""<details class="ai-details"><summary>Lihat Sumber Dokumen ▾</summary><div class="ai-source-list">{links}</div></details>"""
    
    # 3. WAKTU ditaruh SEBELUM SUMBER
    footer_html = ""
    if time_html or sources_html:
        footer_html = f"""<div class="ai-footer">{time_html}{sources_html}</div>"""
    
    return f"{content}\n\n{footer_html}"

# 3. MAIN PAGE FUNCTION

def show_chatbot_page():
    init_user_chat_session()
    
    # --- CSS STYLING ---
    st.markdown("""
        <style>
        /* CSS User Chat & AI Bubble Container */
        .user-chat-container { display: flex; justify-content: flex-end; margin-bottom: 10px; }
        .user-chat-bubble { background-color: #FFB200; color: #FFFFFF; padding: 10px 15px; border-radius: 15px 0px 15px 15px; max-width: 70%; text-align: right; box-shadow: 0px 2px 5px rgba(0,0,0,0.1); font-size: 1rem; }
        div[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] { background-color: #FFFFFF; padding: 15px; border-radius: 0px 15px 15px 15px; border: 1px solid #E0E0E0; box-shadow: 0px 2px 5px rgba(0,0,0,0.05); }

        .ai-footer {
            display: flex;
            flex-direction: column; 
            margin-top: 10px;
            padding-top: 5px;
            border-top: 1px solid #F0F0F0;
        }

        .ai-time {
            align-self: flex-end; 
            font-size: 0.75rem;
            color: #888;
            margin-bottom: 4px; 
        }

        details.ai-details {
            align-self: flex-start;
            width: 100%;
            text-align: left;
        }

        details.ai-details summary {
            list-style: none;
            color: #FFB200;
            font-weight: 600;
            font-size: 0.85rem;
            cursor: pointer;
        }
        
        details.ai-details summary:hover {
            text-decoration: underline;
        }

        .ai-source-list {
            margin-top: 5px;
            padding: 8px;
            background: #FAFAFA;
            border-radius: 5px;
            font-size: 0.8rem;
            border: 1px solid #EEE;
        }
        .ai-source-list a {
            display: block;
            text-decoration: none;
            color: #333;
            margin-bottom: 3px;
        }
        .ai-source-list a:hover {
            color: #FFB200;
        }
        </style>
    """, unsafe_allow_html=True)

    # --- Sidebar ---
    with st.sidebar:
        st.markdown(f"### Selamat Datang, {st.session_state['username']}!")
        st.write("---")
        st.markdown("<div style='flex-grow: 1;'></div>", unsafe_allow_html=True)
        if st.button("Log Out", use_container_width=True, type="primary"):
            st.session_state.logged_in = False
            st.session_state.username = None
            st.session_state.chat_history = []
            st.session_state.conversation_chain = None
            st.session_state.prompt_trigger = None
            st.rerun()
    
    # --- Header ---
    st.markdown("<h1 style='text-align: center;'>DIGICHATBOT</h1>", unsafe_allow_html=True)
    st.markdown("<div style='text-align: center;'>Tanyakan informasi akademik di sini</div>", unsafe_allow_html=True)
    st.write("---")

    # --- FAQ Buttons ---
    if not st.session_state.chat_history:
        st.caption("Pertanyaan yang sering ditanyakan (Klik untuk kirim):")
        col_faq1, col_faq2 = st.columns(2)
        faq_questions = [
            "Gimana cara konversi magang?",
            "Berapa SKS minimal untuk lulus?",
            "Gimana cara daftar sidang akhir?",
            "Gimana cara daftar SUP?",
            "Beasiswa apa saja yang sedang tersedia?",
            "Magang dapat diambil pada semester berapa?",
            "Apa saja syarat daftar wisuda?",
            "Berikan link data prodi"
        ]
        for i, question in enumerate(faq_questions):
            with col_faq1 if i % 2 == 0 else col_faq2:
                if st.button(question, use_container_width=True):
                    st.session_state.prompt_trigger = question
                    st.rerun()
        st.write("")

    # DISPLAY HISTORY
    for message in st.session_state.chat_history:
        if isinstance(message, HumanMessage):
            # Tampilan USER
            st.markdown(f"""
                <div class="user-chat-container">
                    <div class="user-chat-bubble">{message.content}</div>
                </div>
            """, unsafe_allow_html=True)
            
        elif isinstance(message, AIMessage):
            # Tampilan AI
            with st.chat_message("assistant"): 
                r_time = message.metadata.get("response_time", None)
                srcs = message.metadata.get("sources", None)
                final_html = format_ai_message(message.content, r_time, srcs)
                st.markdown(final_html, unsafe_allow_html=True)

    # HANDLING INPUT
    triggered_question = st.session_state.get("prompt_trigger")
    if triggered_question:
        st.session_state.prompt_trigger = None 
    
    chat_input_value = st.chat_input("Ajukan pertanyaan disini...")
    user_question = triggered_question if triggered_question else chat_input_value

    if user_question:
        if st.session_state.conversation_chain is None:
            st.error("Sesi chat tidak terinisialisasi. Coba muat ulang.")
        else:
            # 1. Append User Msg & Tampilkan
            st.session_state.chat_history.append(HumanMessage(content=user_question))
            st.markdown(f"""
                <div class="user-chat-container">
                    <div class="user-chat-bubble">{user_question}</div>
                </div>
            """, unsafe_allow_html=True)
            
            # 2. Proses AI Response
            with st.chat_message("assistant"):
                placeholder = st.empty()
                full_response = ""
                source_documents = []
                response_time = 0.0
                start_time = time.time()

                try:
                    chat_history_for_chain = st.session_state.chat_history[:-1]
                    debug_handler = PrintRetrievalHandler()
                    
                    stream = st.session_state.conversation_chain.stream(
                        {
                        "input": user_question,
                        "chat_history": chat_history_for_chain
                        },
                        config={'callbacks': [debug_handler]}
                    )

                    # A. Streaming Text
                    for chunk in stream:
                        if "answer" in chunk and chunk["answer"]:
                            full_response += chunk["answer"]
                            placeholder.markdown(full_response + "▌")
                        if "context" in chunk:
                            source_documents = chunk["context"]
                    
                    end_time = time.time()
                    response_time = end_time - start_time
                    
                    # B. Proses Sumber Dokumen
                    ai_sources = {}
                    if source_documents:
                        sources = {doc.metadata.get('source', None) for doc in source_documents}
                        sources = {s for s in sources if s}
                        if sources:
                            try:
                                supabase = st.session_state['supabase']
                                for source_file in sources:
                                    # Dapatkan URL publik
                                    public_url = supabase.storage.from_('pdf_documents').get_public_url(source_file)
                                    ai_sources[source_file] = public_url
                            except Exception as e:
                                print(f"Gagal mengambil URL gambar: {e}")
                                pass

                    # C. Render Final
                    final_html_display = format_ai_message(full_response, response_time, ai_sources)
                    placeholder.markdown(final_html_display, unsafe_allow_html=True)

                except Exception as e:
                    st.error(f"Terjadi kesalahan: {e}")
                    full_response = "Maaf, saya mengalami gangguan saat memproses data."
                    placeholder.markdown(full_response)
                    response_time = 0.0
                    ai_sources = {}

                # 3. Simpan Log & Update History
                save_chat_log(
                    st.session_state.get('username', 'Anonymous'),
                    user_question, full_response, response_time
                )
                
                st.session_state.chat_history.append(
                    AIMessage(
                        content=full_response, 
                        metadata={"sources": ai_sources, "response_time": response_time} 
                    )
                )