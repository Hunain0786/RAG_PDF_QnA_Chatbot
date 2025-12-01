import os
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# -----------------------------
# INITIALIZE SERVICES
# -----------------------------

# Embedding model
embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Pinecone init
from dotenv import load_dotenv

load_dotenv()

# Pinecone init
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

index_name = "pdf-rag-index"

if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

index = pc.Index(index_name)

# Flan-T5 model
tok = AutoTokenizer.from_pretrained("google/flan-t5-base")
llm = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")
