import streamlit as st
import sys
import os
sys.path.insert(0, os.path.abspath("."))

st.set_page_config(page_title="FinSight", page_icon="🏦")
st.title("🏦 FinSight")
st.write("App started successfully")
st.write(f"Working directory: {os.getcwd()}")

query = st.text_input("Enter query")

if st.button("Search"):
    st.write("Button clicked")

    try:
        st.write("Step 1: importing chromadb...")
        import chromadb
        st.write("✓ chromadb imported")

        st.write("Step 2: connecting to ChromaDB...")
        chroma_path = os.path.join(os.getcwd(), "data", "chroma_db")
        st.write(f"Path: {chroma_path}")
        st.write(f"Exists: {os.path.exists(chroma_path)}")
        client = chromadb.PersistentClient(path=chroma_path)
        collection = client.get_collection("finsight")
        st.write(f"✓ ChromaDB loaded — {collection.count()} chunks")

        st.write("Step 3: loading embedding model...")
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        st.write("✓ Embedding model loaded")

        st.write("Step 4: loading cross-encoder...")
        from sentence_transformers import CrossEncoder
        ce = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        st.write("✓ Cross-encoder loaded")

        st.write("Step 5: running full pipeline...")
        from src.retrieval import hybrid_search
        from src.reranker import rerank
        from src.generate import generate_answer
        candidates = hybrid_search(query, None, None, top_k=20)
        reranked = rerank(query, candidates, top_k=5)
        answer = generate_answer(query, reranked)
        st.write("✓ Pipeline complete")
        st.markdown("### Answer")
        st.markdown(answer)

    except Exception as e:
        import traceback
        st.error(f"Error: {e}")
        st.code(traceback.format_exc())
