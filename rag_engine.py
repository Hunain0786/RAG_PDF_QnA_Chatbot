import numpy as np
from config import embedder, index, groq_client


def retrieve_chunks(query, top_k=5):
    query_emb = embedder.encode(query).astype(np.float32).tolist()

    response = index.query(
        vector=query_emb,
        top_k=top_k,
        include_metadata=True
    )

    chunks = [m["metadata"]["text"] for m in response["matches"]]
    return chunks


def build_prompt(query, chunks):
    """Build a prompt with retrieved context for the LLM."""
    context = "\n\n".join(chunks)
    
    prompt = (
        "You are an expert AI assistant. Use the following context extracted from a PDF to answer the user's question.\n\n"
        "### Guidelines:\n"
        "1. If the user asks for a summary or an explanation of the document, provide a comprehensive overview based on the provided context.\n"
        "2. If the user greets you, respond politely.\n"
        "3. If the answer is not in the context and it's a specific fact-based question, say 'I don't know.'\n"
        "4. If the question is general (how are you, etc.), respond politely without needing context.\n"
        "5. Always be concise but informative.\n\n"
        f"### Context:\n{context}\n\n"
        f"### User Question:\n{query}\n\n"
        "### AI Response:"
    )
    
    return prompt


def is_global_query(query):
    """Check if the query is asking for a general explanation or summary."""
    global_keywords = ["summarize", "summary", "explain", "about", "what is this", "overview", "tell me about"]
    query_lower = query.lower()
    return any(keyword in query_lower for keyword in global_keywords)


def answer_question(query, model="llama-3.3-70b-versatile"):
    """
    Answer a question using RAG with Groq.
    """
    # If it's a global query, retrieve more chunks to get a better overview
    top_k = 20 if is_global_query(query) else 5
    
    chunks = retrieve_chunks(query, top_k=top_k)
    prompt = build_prompt(query, chunks)
    
    # Call Groq API
    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant that answers questions based on the provided context. If the user asks to explain or summarize, use the provided context to give a high-level overview."
        },
        {
            "role": "user",
            "content": prompt
        }
    ]
    
    response = groq_client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.5,
        max_tokens=1000
    )
    
    return response.choices[0].message.content
