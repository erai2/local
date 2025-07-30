from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.document_loaders import (
    TextLoader,
    PyPDFLoader,
    UnstructuredWordDocumentLoader
)

import os
import glob

def load_documents(directory):
    docs = []
    for path in glob.glob(f"{directory}/*"):
        if path.endswith(".txt"):
            loader = TextLoader(path, encoding="utf-8")
        elif path.endswith(".pdf"):
            loader = PyPDFLoader(path)
        elif path.endswith(".docx") or path.endswith(".doc"):
            loader = UnstructuredWordDocumentLoader(path)
        else:
            continue
        docs.extend(loader.load())
    return docs

def build_chroma_db(doc_dir="backend/uploaded_docs", persist_dir="backend/chroma_db"):
    documents = load_documents(doc_dir)
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = splitter.split_documents(documents)
    embedding = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    Chroma.from_documents(texts, embedding=embedding, persist_directory=persist_dir)

def get_answer_from_documents(question, doc_dir="backend/uploaded_docs", persist_dir="backend/chroma_db"):
    if not os.path.exists(persist_dir):
        build_chroma_db(doc_dir, persist_dir)
    embedding = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    vectordb = Chroma(persist_directory=persist_dir, embedding_function=embedding)
    retriever = vectordb.as_retriever(search_kwargs={"k": 3})
    qa_chain = RetrievalQA.from_chain_type(llm=ChatOpenAI(temperature=0), retriever=retriever)
    return qa_chain.run(question)
