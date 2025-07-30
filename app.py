import streamlit as st
import os
import shutil
from langchain_community.document_loaders import PyPDFLoader, UnstructuredWordDocumentLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from sklearn.cluster import KMeans
from sentence_transformers import SentenceTransformer
import pandas as pd

# --- 0. Basic Settings ---
st.set_page_config(page_title="통합 문서 분석 시스템", layout="wide")
st.title("🧩 통합 문서 분석 및 RAG 시스템")

# Directory to save uploaded files
UPLOAD_DIR = "./uploaded_docs"
if not os.path.exists(UPLOAD_DIR):
    os.makedirs(UPLOAD_DIR)

# --- 1. Core Logic Functions (Integrating previous files) ---

@st.cache_data
def load_documents(path_or_directory):
    """
    Loads all documents from a directory if the path is a directory,
    or loads a single file if the path is a file.
    """
    docs = []
    paths_to_load = []

    if os.path.isdir(path_or_directory):
        for filename in os.listdir(path_or_directory):
            paths_to_load.append(os.path.join(path_or_directory, filename))
    elif os.path.isfile(path_or_directory):
        paths_to_load.append(path_or_directory)

    for path in paths_to_load:
        filename = os.path.basename(path)
        try:
            if filename.endswith(".pdf"):
                loader = PyPDFLoader(path)
            elif filename.endswith(".docx") or filename.endswith(".doc"):
                loader = UnstructuredWordDocumentLoader(path)
            elif filename.endswith(".txt"):
                loader = TextLoader(path, encoding="utf-8")
            else:
                continue
            docs.extend(loader.load())
        except Exception as e:
            # Display a warning but continue execution
            st.warning(f"'{filename}' 파일 로딩 중 오류 발생: {e}")
    return docs

@st.cache_resource
def build_rag_chain(_docs, openai_api_key):
    """Builds the RAG chain."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    splits = text_splitter.split_documents(_docs)

    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    vectorstore = FAISS.from_documents(documents=splits, embedding=embeddings)

    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=ChatOpenAI(temperature=0, openai_api_key=openai_api_key),
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return qa_chain

@st.cache_data
def summarize_text(text, openai_api_key, model="gpt-3.5-turbo"):
    """Summarizes document content using AI."""
    client = ChatOpenAI(temperature=0, openai_api_key=openai_api_key, model_name=model)
    prompt = f"다음 텍스트를 핵심 내용만 간추려 한국어로 명확하게 요약해줘:\n\n{text[:4000]}"
    summary = client.invoke(prompt)
    return summary.content

# --- 2. Streamlit UI Configuration ---

# Sidebar: Settings and File Management
with st.sidebar:
    st.header("⚙️ 설정")
    if 'OPENAI_API_KEY' in st.secrets:
        openai_api_key = st.secrets['OPENAI_API_KEY']
        st.success("API Key가 안전하게 로드되었습니다.")
    else:
        openai_api_key = st.text_input("OpenAI API Key", type="password")
        if not openai_api_key:
            st.warning("OpenAI API 키를 입력해주세요.")

    st.header("📂 문서 관리")
    uploaded_file = st.file_uploader("문서 업로드", accept_multiple_files=False)
    if uploaded_file:
        file_path = os.path.join(UPLOAD_DIR, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.success(f"'{uploaded_file.name}' 업로드 완료!")
        st.rerun()

    files = sorted(os.listdir(UPLOAD_DIR))
    if files:
        selected_file_for_delete = st.selectbox("삭제할 파일 선택", options=[""] + files)
        if selected_file_for_delete and st.button("선택한 파일 삭제"):
            os.remove(os.path.join(UPLOAD_DIR, selected_file_for_delete))
            st.success(f"'{selected_file_for_delete}' 삭제 완료!")
            st.rerun()
    else:
        st.info("업로드된 문서가 없습니다.")

# Main Screen: Tabbed Interface for Features
tab1, tab2, tab3 = st.tabs(["💬 문서 기반 Q&A (RAG)", "✍️ 문서 요약", "📊 문서 군집 분석"])

# --- Tab 1: RAG Q&A ---
with tab1:
    st.subheader("AI에게 문서에 대해 질문하세요")
    if not openai_api_key:
        st.warning("사이드바에서 OpenAI API 키를 먼저 입력해주세요.")
    elif not files:
        st.info("질문할 문서를 먼저 업로드해주세요.")
    else:
        if "rag_chain" not in st.session_state or st.button("문서 변경, 체인 재생성"):
            with st.spinner("문서를 분석하여 RAG 체인을 빌드하는 중..."):
                docs = load_documents(UPLOAD_DIR)
                if docs:
                    st.session_state.rag_chain = build_rag_chain(docs, openai_api_key)
                    st.success("RAG 체인 빌드 완료!")
                else:
                    st.error("문서 로딩에 실패했습니다. 지원하는 형식의 파일인지 확인해주세요.")
        
        if "messages" not in st.session_state:
            st.session_state.messages = []

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        if prompt := st.chat_input("질문을 입력하세요..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                with st.spinner("답변 생성 중..."):
                    if "rag_chain" in st.session_state:
                        response = st.session_state.rag_chain({"question": prompt})
                        answer = response['answer']
                        st.markdown(answer)
                        st.session_state.messages.append({"role": "assistant", "content": answer})
                    else:
                        st.error("RAG 체인이 초기화되지 않았습니다.")

# --- Tab 2: Document Summary ---
with tab2:
    st.subheader("선택한 문서를 AI가 요약합니다")
    if not openai_api_key:
        st.warning("사이드바에서 OpenAI API 키를 먼저 입력해주세요.")
    elif not files:
        st.info("요약할 문서를 먼저 업로드해주세요.")
    else:
        selected_file_for_summary = st.selectbox("요약할 파일 선택", options=[""] + files, key="summary_select")
        if selected_file_for_summary and st.button("선택한 파일 요약하기"):
            with st.spinner(f"'{selected_file_for_summary}' 파일 요약 중..."):
                file_path = os.path.join(UPLOAD_DIR, selected_file_for_summary)
                docs = load_documents(file_path)
                
                # ✅ ERROR FIX: Check if the docs list is not empty before accessing it
                if docs:
                    summary = summarize_text(docs[0].page_content, openai_api_key)
                    st.success("요약 결과:")
                    st.write(summary)
                else:
                    st.error("문서 내용을 읽을 수 없습니다. 파일이 손상되었거나 지원하지 않는 형식일 수 있습니다.")

# --- Tab 3: Document Clustering ---
with tab3:
    st.subheader("업로드된 모든 문서를 내용 기반으로 그룹화합니다")
    if not files or len(files) < 2:
        st.info("분석하려면 2개 이상의 문서를 업로드해주세요.")
    else:
        if st.button("전체 문서 분석 및 군집화 실행"):
            with st.spinner("모든 문서를 벡터화하고 군집 분석을 수행하는 중..."):
                docs_for_cluster = []
                for f in files:
                    file_path = os.path.join(UPLOAD_DIR, f)
                    loaded_docs = load_documents(file_path)
                    
                    # ✅ ERROR FIX: Check if the loaded_docs list is not empty
                    if loaded_docs:
                        docs_for_cluster.append({"filename": f, "text": loaded_docs[0].page_content})
                
                if len(docs_for_cluster) >= 2:
                    texts = [d['text'] for d in docs_for_cluster]
                    model = SentenceTransformer("all-MiniLM-L6-v2")
                    embeddings = model.encode(texts)
                    
                    num_clusters = min(len(docs_for_cluster), 4)
                    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init='auto').fit(embeddings)
                    
                    result_df = pd.DataFrame({
                        "파일명": [d['filename'] for d in docs_for_cluster],
                        "그룹 번호": kmeans.labels_
                    })
                    
                    st.success("군집 분석 결과:")
                    st.dataframe(result_df.sort_values(by="그룹 번호").reset_index(drop=True))
                else:
                    st.error("분석할 수 있는 텍스트를 가진 문서가 2개 미만입니다.")
