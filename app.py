"""
RAG API with FastAPI + LangChain (LCEL) + Mistral AI + built-in HTML UI
Run: uvicorn app:app --reload
Then open: http://localhost:8000/
"""

import hashlib
import os
import shutil
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import BackgroundTasks, FastAPI, File, HTTPException, Request, UploadFile
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from langchain_mistralai import ChatMistralAI, MistralAIEmbeddings
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

load_dotenv()

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Tuning knobs for large PDFs
CHUNK_SIZE = 800          # bigger chunks -> fewer embedding calls for huge books
CHUNK_OVERLAP = 100
EMBED_BATCH_SIZE = 64     # texts per embedding API call
MAX_WORKERS = 8           # parallel embedding batches in flight

state = {}
jobs = {}  # job_id -> {"status", "progress", "total", "filename", "error"}

# documents: filename -> {
#   "current_hash": str,
#   "versions": [{"hash", "version", "uploaded_at", "path"}, ...]   # oldest -> newest
# }
documents = {}

PROMPT = ChatPromptTemplate.from_template("""
Answer the question using ONLY the context below.
If the answer isn't in the context, say you don't know.

Context:
{context}

Question: {input}
""")


def hash_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def check_upload_status(filename: str, file_hash: str):
    """Returns one of: 'new_doc', 'duplicate', 'new_version'."""
    doc = documents.get(filename)
    if doc is None:
        return "new_doc"
    if doc["current_hash"] == file_hash:
        return "duplicate"
    # same filename, different content -> new version
    return "new_version"


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def load_and_split(pdf_path: str):
    # PyMuPDF is dramatically faster than pypdf for large/scanned-text PDFs
    loader = PyMuPDFLoader(pdf_path)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )
    return splitter.split_documents(docs)


def embed_batch(embeddings, texts):
    return embeddings.embed_documents(texts)


def build_index_parallel(chunks, embeddings, job_id: str):
    """Embed chunks in parallel batches and assemble a FAISS index incrementally."""
    total = len(chunks)
    jobs[job_id]["total"] = total
    jobs[job_id]["progress"] = 0

    batches = [chunks[i : i + EMBED_BATCH_SIZE] for i in range(0, total, EMBED_BATCH_SIZE)]
    results = [None] * len(batches)

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        future_to_idx = {
            pool.submit(embed_batch, embeddings, [d.page_content for d in batch]): idx
            for idx, batch in enumerate(batches)
        }
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            results[idx] = future.result()
            jobs[job_id]["progress"] += len(batches[idx])

    vectorstore = None
    for idx, batch in enumerate(batches):
        texts = [d.page_content for d in batch]
        metadatas = [d.metadata for d in batch]
        text_embeddings = list(zip(texts, results[idx]))

        if vectorstore is None:
            vectorstore = FAISS.from_embeddings(
                text_embeddings, embeddings, metadatas=metadatas
            )
        else:
            vectorstore.add_embeddings(text_embeddings, metadatas=metadatas)

    return vectorstore


def process_pdf_job(job_id: str, pdf_path: str, filename: str, file_hash: str):
    try:
        jobs[job_id]["status"] = "splitting"
        chunks = load_and_split(pdf_path)

        jobs[job_id]["status"] = "embedding"
        embeddings = state["embeddings"]
        vectorstore = build_index_parallel(chunks, embeddings, job_id)

        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
        rag_chain = (
            {"context": retriever | format_docs, "input": RunnablePassthrough()}
            | PROMPT
            | state["llm"]
            | StrOutputParser()
        )

        state["rag_chain"] = rag_chain
        state["retriever"] = retriever
        state["active_pdf"] = filename

        doc = documents.setdefault(filename, {"current_hash": None, "versions": []})
        next_version = len(doc["versions"]) + 1
        doc["versions"].append({
            "hash": file_hash,
            "version": next_version,
            "uploaded_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "path": pdf_path,
        })
        doc["current_hash"] = file_hash

        jobs[job_id]["status"] = "done"
        jobs[job_id]["version"] = next_version
    except Exception as e:
        jobs[job_id]["status"] = "error"
        jobs[job_id]["error"] = str(e)


@asynccontextmanager
async def lifespan(app: FastAPI):
    state["embeddings"] = MistralAIEmbeddings(model="mistral-embed")
    state["llm"] = ChatMistralAI(model="mistral-large-latest", temperature=0.3)
    print("Embeddings + LLM ready. Upload a PDF to start querying.")

    yield

    state.clear()


app = FastAPI(title="RAG API", lifespan=lifespan)

# --- UI wiring ---
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


@app.get("/")
def home(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")


# --- API models ---
class QueryRequest(BaseModel):
    question: str


class SourceChunk(BaseModel):
    content: str
    page: int | None = None


class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceChunk]


@app.get("/health")
def health():
    active_pdf = state.get("active_pdf")
    active_version = None
    if active_pdf and active_pdf in documents:
        active_version = documents[active_pdf]["versions"][-1]["version"]
    return {
        "status": "ok",
        "ready": "rag_chain" in state,
        "active_pdf": active_pdf,
        "active_version": active_version,
    }


@app.post("/upload")
async def upload_pdf(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    force: bool = False,
):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    safe_name = f"{uuid.uuid4().hex}_{file.filename}"
    save_path = os.path.join(UPLOAD_DIR, safe_name)

    try:
        with open(save_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
    finally:
        file.file.close()

    file_hash = hash_file(save_path)
    status = check_upload_status(file.filename, file_hash)

    if status == "duplicate" and not force:
        os.remove(save_path)
        existing_version = documents[file.filename]["versions"][-1]["version"]
        return {
            "result": "duplicate",
            "filename": file.filename,
            "existing_version": existing_version,
        }

    job_id = uuid.uuid4().hex
    jobs[job_id] = {
        "status": "queued",
        "progress": 0,
        "total": 0,
        "filename": file.filename,
        "error": None,
        "version": None,
        "is_new_version": status == "new_version",
    }

    background_tasks.add_task(process_pdf_job, job_id, save_path, file.filename, file_hash)

    return {"result": status, "job_id": job_id, "filename": file.filename}


@app.get("/documents")
def list_documents():
    return {
        name: {
            "current_version": doc["versions"][-1]["version"],
            "version_count": len(doc["versions"]),
            "versions": [
                {"version": v["version"], "uploaded_at": v["uploaded_at"]}
                for v in doc["versions"]
            ],
        }
        for name, doc in documents.items()
    }


@app.get("/upload/status/{job_id}")
def upload_status(job_id: str):
    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Unknown job id")
    return job


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest):
    if "rag_chain" not in state:
        raise HTTPException(status_code=503, detail="RAG pipeline not initialized")

    try:
        answer = state["rag_chain"].invoke(request.question)
        retrieved_docs = state["retriever"].invoke(request.question)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"RAG pipeline error: {e}")

    sources = [
        SourceChunk(content=doc.page_content[:300], page=doc.metadata.get("page"))
        for doc in retrieved_docs
    ]

    return QueryResponse(answer=answer, sources=sources)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)