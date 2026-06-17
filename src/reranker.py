from sentence_transformers import CrossEncoder
import streamlit as st

@st.cache_resource
def get_cross_encoder():
    return CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

def rerank(query, hybrid_results, top_k=5):
    if not hybrid_results:
        return []
    cross_encoder = get_cross_encoder()
    pairs = [[query, doc] for (_, doc, _) in hybrid_results]
    scores = cross_encoder.predict(pairs)
    ranked = sorted(
        zip(hybrid_results, scores),
        key=lambda x: x[1],
        reverse=True
    )
    return [(chunk_id, doc, meta, float(score))
            for (chunk_id, doc, meta), score in ranked[:top_k]]

def query_pipeline(query, company=None, year=None, top_k=5):
    from src.retrieval import hybrid_search
    candidates = hybrid_search(query, company, year, top_k=20)
    reranked = rerank(query, candidates, top_k=top_k)
    return reranked