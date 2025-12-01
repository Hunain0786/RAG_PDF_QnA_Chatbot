import numpy as np
from config import embedder, index, tok, llm

# -----------------------------
# PINECONE RETRIEVAL
# -----------------------------

def retrieve_chunks(query, top_k=5):
    query_emb = embedder.encode(query).astype(np.float32).tolist()

    response = index.query(
        vector=query_emb,
        top_k=top_k,
        include_metadata=True
    )

    chunks = [m["metadata"]["text"] for m in response["matches"]]
    return chunks


# -----------------------------
# PROMPT + LLM
# -----------------------------

def build_safe_prompt(query, chunks, max_input_tokens=512):

    base = (
        "Use the context to answer the question in a complete sentence.\n"
        "Provide an explanation if possible.\n"
        "If the answer is not in the context, say 'I don't know.'\n\n"
        "Context:\n"
    )

    used = []
    for c in chunks:
        temp_context = "\n\n".join(used + [c])
        temp_prompt = f"{base}{temp_context}\n\nQuestion: {query}\nAnswer:"

        tok_len = len(tok(temp_prompt)["input_ids"])

        if tok_len <= max_input_tokens - 50:
            used.append(c)
        else:
            break

    final_context = "\n\n".join(used)
    final_prompt = f"{base}{final_context}\n\nQuestion: {query}\nAnswer:"

    return final_prompt


def answer_question(query):
    chunks = retrieve_chunks(query, top_k=5)
    prompt = build_safe_prompt(query, chunks, max_input_tokens=512)

    tokens = tok(prompt, return_tensors="pt", truncation=True, max_length=512)

    output = llm.generate(
        **tokens,
        max_new_tokens=150,
        do_sample=False
    )

    return tok.decode(output[0], skip_special_tokens=True)
