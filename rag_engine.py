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
        "Use the context to answer the question in a complete sentence.\n"
        "Provide an explanation if possible.\n"
        "If the answer is not in the context, say 'I don't know.'\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {query}\n"
        f"Answer:"
    )
    
    return prompt


def answer_question(query, model="llama-3.3-70b-versatile"):
    """
    Answer a question using RAG with Groq.
    
    Args:
        query: The user's question
        model: Groq model to use (default: llama-3.3-70b-versatile)
               Other options: mixtral-8x7b-32768, llama-3.1-8b-instant, etc.
    """
    chunks = retrieve_chunks(query, top_k=5)
    prompt = build_prompt(query, chunks)
    
    # Call Groq API
    response = groq_client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant that answers questions based on the provided context."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.3,
        max_tokens=500
    )
    
    return response.choices[0].message.content
