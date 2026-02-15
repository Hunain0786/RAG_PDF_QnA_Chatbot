import os
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer


embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

from dotenv import load_dotenv

load_dotenv()

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

from groq import Groq
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
