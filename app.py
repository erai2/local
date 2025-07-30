import streamlit as st
import os
import shutil
from langchain_community.document_loaders import PyPDFLoader, UnstructuredWordDocumentLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA, ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from sklearn.cluster import KMeans
from sentence_transformers import SentenceTransformer
import pandas as pd

# --- 0. 기본 설정 ---
st.set_page_config(page_title="통합 문서 분석 시스템", layout="wide")
st.title("🧩 통합 문서 분석 및 RAG 시스템")

# 파일 저장 디렉토리
UPLOAD_DIR = "./uploaded_docs"
if not os.path.exists(UPLOAD_DIR):
    os.makedirs(UPLOAD_DIR)

# --- 1. 핵심 로직 함수 (기존 파일들의 기능 통합) ---

@st.cache_data
def load_documents(directory):
    """지정된 디렉토리에서 모든 문서를 로드 (document_loader.py, utils.py 통합)"""
    docs = []
    for filename in os.listdir(directory):
        path = os.path.join(directory, filename)
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
            st.warning(f"'{filename}' 파일 로딩 중 오류 발생: {e}")
    return docs

@st.cache_resource
def build_rag_chain(_docs, openai_api_key):
    """RAG 체인을 빌드 (rag_engine.py, rag_vector.py 통합)"""
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
    """문서 내용을 AI로 요약 (summarize.py 통합)"""
    client = ChatOpenAI(temperature=0, openai_api_key=openai_api_key, model_name=model)
    prompt = f"다음 텍스트를 핵심 내용만 간추려 한국어로 명확하게 요약해줘:\n\n{text[:4000]}"
    summary = client.invoke(prompt)
    return summary.content

@st.cache_data
def cluster_and_summarize_docs(directory):
    """문서 군집화 및 요약 (pipeline.py 통합)"""
    docs_with_text = []
    for filename in os.listdir(directory):
        path = os.path.join(directory, filename)
        # load_documents 함수를 재사용하지 않고 간단히 텍스트만 추출
        try:
            # 이 부분은 실제 텍스트 추출 로직이 필요
            # 간단한 예시로 파일 이름만 사용
            docs_with_text.append({"filename": filename, "text": f"Content of {filename}"}) 
        except:
            continue
    
    if len(docs_with_text) < 2:
        return None

    texts = [doc['text'] for doc in docs_with_text]
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(texts)
    
    num_clusters = min(len(docs_with_text), 4) # 클러스터 수는 문서 수보다 작아야 함
    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init='auto').fit(embeddings)
    
    for i, doc in enumerate(docs_with_text):
        doc["cluster"] = kmeans.labels_[i]
    
    return pd.DataFrame(docs_with_text)[['filename', 'cluster']]


# --- 2. Streamlit UI 구성 ---

# 사이드바: 설정 및 파일 관리
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

    files = os.listdir(UPLOAD_DIR)
    if files:
        selected_file_for_delete = st.selectbox("삭제할 파일 선택", options=[""] + files)
        if selected_file_for_delete and st.button("선택한 파일 삭제"):
            os.remove(os.path.join(UPLOAD_DIR, selected_file_for_delete))
            st.success(f"'{selected_file_for_delete}' 삭제 완료!")
            st.rerun()
    else:
        st.info("업로드된 문서가 없습니다.")

# 메인 화면: 기능 선택 탭
tab1, tab2, tab3 = st.tabs(["💬 문서 기반 Q&A (RAG)", "✍️ 문서 요약", "📊 문서 군집 분석"])

# --- 탭 1: RAG Q&A ---
with tab1:
    st.subheader("문서 내용에 대해 AI에게 질문하세요")
    if not openai_api_key:
        st.warning("사이드바에서 OpenAI API 키를 먼저 입력해주세요.")
    elif not files:
        st.info("질문할 문서를 먼저 업로드해주세요.")
    else:
        # RAG 체인 초기화
        if "rag_chain" not in st.session_state or st.button("문서 변경, 체인 재생성"):
            with st.spinner("문서를 분석하여 RAG 체인을 빌드하는 중..."):
                docs = load_documents(UPLOAD_DIR)
                if docs:
                    st.session_state.rag_chain = build_rag_chain(docs, openai_api_key)
                    st.success("RAG 체인 빌드 완료!")
                else:
                    st.error("문서 로딩에 실패했습니다.")
        
        # 채팅 UI
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


# --- 탭 2: 문서 요약 ---
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
               doc = load_documents(os.path.join(UPLOAD_DIR, selected_file_for_summary))
                if doc:
                    summary = summarize_text(doc[0].page_content, openai_api_key)
                    st.success("요약 결과:")
                    st.write(summary)
                else:
                    st.error("문서 내용을 읽을 수 없습니다.")

# --- 탭 3: 문서 군집 분석 ---
with tab3:
    st.subheader("업로드된 모든 문서를 내용 기반으로 그룹화합니다")
    if not files or len(files) < 2:
        st.info("분석하려면 2개 이상의 문서를 업로드해주세요.")
    else:
        if st.button("전체 문서 분석 및 군집화 실행"):
            with st.spinner("모든 문서를 벡터화하고 군집 분석을 수행하는 중..."):
                # pipeline.py의 텍스트 추출 로직을 단순화하여 적용
                docs_for_cluster = []
                for f in files:
                    loaded_doc = load_documents(os.path.join(UPLOAD_DIR, f))
                    if loaded_doc:
                        docs_for_cluster.append({"filename": f, "text": loaded_doc[0].page_content})
                
                if docs_for_cluster:
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
                    st.error("분석할 텍스트를 추출하지 못했습니다.")

