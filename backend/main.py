
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import os, shutil
from rag_vector import get_answer_from_documents

UPLOAD_DIR = "backend/uploaded_docs"
os.makedirs(UPLOAD_DIR, exist_ok=True)

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    return {"status": "ok"}

@app.get("/files")
def list_files():
    return {"files": os.listdir(UPLOAD_DIR)}

@app.delete("/delete/{filename}")
def delete_file(filename: str):
    path = os.path.join(UPLOAD_DIR, filename)
    if os.path.exists(path):
        os.remove(path)
        return {"status": "deleted"}
    return {"error": "file not found"}

@app.post("/query")
async def query(data: dict):
    question = data.get("question", "")
    answer = get_answer_from_documents(question, UPLOAD_DIR)
    return {"answer": answer}
