
import os
import glob
from langchain.document_loaders import TextLoader, PyPDFLoader, UnstructuredWordDocumentLoader

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
