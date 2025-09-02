import streamlit as st
from rag_pipeline import ask_question

st.set_page_config(page_title="📚 Mathematical Statistics RAG Chatbot", layout="wide")

st.title("📚 Mathematical Statistics RAG Chatbot")
st.markdown("Ask me anything about Statistics If it is in my knowledgebase i give you answer. (Powered by Gemini + Pinecone)")

user_query = st.text_input("💬 Enter your question:")

if st.button("Ask"):
    if user_query.strip():
        with st.spinner("🔍 Processing your query..."):
            try:
                response, sources = ask_question(user_query, top_k=3)

                st.subheader("🤖 Answer")
                st.write(response)

                st.subheader("📖 Relevant Sources")
                for i, chunk in enumerate(sources, 1):
                    with st.expander(f"Source {i} (Similarity: {chunk['score']:.3f})"):
                        st.write(chunk["text"])
            except Exception as e:
                st.error(f"❌ Error: {e}")
    else:
        st.warning("Please enter a question first.")
