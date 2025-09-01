import google.generativeai as genai
from pinecone import Pinecone

GEMINI_API_KEY = "AIzaSyB5CxJCnrgbtVaEIdZj4OGGV_EO82B-X3w"
PINECONE_API_KEY = "pcsk_2avFR6_EVP7EQnoTGXhNKNWxByYaM8WYgGsrM9qQ9SJyPNBzJuAbecgaoSFHPchvXSkC89"
PINECONE_INDEX_NAME = "nitinr"
# ---------------------------------------------------
# 1. Create RAG System
# ---------------------------------------------------
def create_rag_system():
    pc = Pinecone(api_key=PINECONE_API_KEY)
    pinecone_index = pc.Index(PINECONE_INDEX_NAME)
    genai.configure(api_key=GEMINI_API_KEY)
    return pinecone_index


# ---------------------------------------------------
# 2. Embedding
# ---------------------------------------------------
def embed_query(query_text):
    response = genai.embed_content(
        model="models/text-embedding-004",
        content=query_text
    )
    return response["embedding"]


# ---------------------------------------------------
# 3. Retriever
# ---------------------------------------------------
def retrieve_relevant_chunks(pinecone_index, query_embedding, top_k=5):
    search_results = pinecone_index.query(
        vector=query_embedding,
        top_k=top_k,
        include_metadata=True
    )

    relevant_chunks = []
    for match in search_results.matches:
        chunk_text = match.metadata["text"]
        similarity_score = match.score
        relevant_chunks.append({
            "text": chunk_text,
            "score": similarity_score
        })
    return relevant_chunks


# ---------------------------------------------------
# 4. Generator
# ---------------------------------------------------
def generate_rag_response(query, relevant_chunks):
    context = "\n\n".join([chunk["text"] for chunk in relevant_chunks])

    rag_prompt = f"""
You have to behave like a Mathematics and Statistics Expert
You will be given a context of relevant information and a user question.
Context:
{context}

Question: {query}

Instructions:
1. Answer only from the context.
2. If not enough info, say "I don’t have enough information in the provided context."
3.  Keep your answers clear, concise, and educational.
Answer:
"""

    # ✅ Correct usage for new SDK
    model = genai.GenerativeModel("gemini-2.0-flash")
    response = model.generate_content(rag_prompt)
    return response.text


# ---------------------------------------------------
# 5. Full Pipeline
# ---------------------------------------------------
def ask_question(query, top_k=5):
    pinecone_index = create_rag_system()
    query_embedding = embed_query(query)
    relevant_chunks = retrieve_relevant_chunks(pinecone_index, query_embedding, top_k)
    response = generate_rag_response(query, relevant_chunks)
    return response, relevant_chunks
