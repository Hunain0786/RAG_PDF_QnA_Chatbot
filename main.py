import uuid
import numpy as np
from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel

from config import embedder, index
from pdf_processor import extract_text, chunk_text
from rag_engine import answer_question

# -----------------------------
# INITIALIZE SERVICES
# -----------------------------

app = FastAPI()

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# FASTAPI ENDPOINTS
# -----------------------------

@app.post("/upload_pdf")
async def upload_pdf(pdf: UploadFile = File(...)):
    """Upload a PDF and index it into Pinecone."""

    pdf_path = f"./{pdf.filename}"
    with open(pdf_path, "wb") as f:
        f.write(await pdf.read())

    text = extract_text(pdf_path)

    chunks = chunk_text(text)

    embeddings = embedder.encode(chunks).astype(np.float32)

    vectors = []
    for i, emb in enumerate(embeddings):
        vectors.append({
            "id": str(uuid.uuid4()),
            "values": emb.tolist(),
            "metadata": {"text": chunks[i]}
        })

    index.delete(delete_all=True)

    index.upsert(vectors)

    return {
        "message": "PDF processed successfully",
        "chunks_indexed": len(chunks)
    }


class QuestionRequest(BaseModel):
    query: str


@app.post("/ask")
async def ask_question(req: QuestionRequest):
    """Ask a question based on the indexed PDF."""
    answer = answer_question(req.query)
    return {"answer": answer}


@app.get("/")
def home():
    return {"message": "PDF RAG Chatbot API is running!"}
