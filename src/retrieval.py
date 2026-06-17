import streamlit as st
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
import os

@st.cache_resource
def load_store():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(base_dir, "data", "vector_store.json")
    with open(path, "r") as f:
        data = json.load(f)
    embeddings = np.array(data["embeddings"], dtype=np.float32)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / np.where(norms == 0, 1, norms)
    return {
        "ids": data["ids"],
        "documents": data["documents"],
        "metadatas": data["metadatas"],
        "embeddings": embeddings
    }

@st.cache_resource
def get_model():
    return SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

@st.cache_resource
def get_bm25():
    store = load_store()
    corpus = [doc.lower().split() for doc in store["documents"]]
    bm25 = BM25Okapi(corpus)
    return bm25


def _filter_mask(metadatas, company, year):
    mask = np.ones(len(metadatas), dtype=bool)
    if company:
        mask &= np.array([m["company"] == company for m in metadatas])
    if year:
        mask &= np.array([m["year"] == str(year) for m in metadatas])
    return mask


def semantic_search(query, company=None, year=None, n=20):
    model = get_model()
    store = load_store()
    q_emb = model.encode([query])[0]
    q_emb = q_emb / (np.linalg.norm(q_emb) or 1)

    mask = _filter_mask(store["metadatas"], company, year)
    idxs = np.where(mask)[0]
    if len(idxs) == 0:
        return [], [], []

    sims = store["embeddings"][idxs] @ q_emb
    top_local = np.argsort(sims)[::-1][:n]
    top_idxs = idxs[top_local]

    ids = [store["ids"][i] for i in top_idxs]
    docs = [store["documents"][i] for i in top_idxs]
    metas = [store["metadatas"][i] for i in top_idxs]
    return ids, docs, metas


def bm25_search(query, company=None, year=None, n=20):
    bm25 = get_bm25()
    store = load_store()
    tokenized = query.lower().split()
    scores = bm25.get_scores(tokenized)

    mask = _filter_mask(store["metadatas"], company, year)
    idxs = np.where(mask)[0]
    filtered_scores = [(i, scores[i]) for i in idxs]
    filtered_scores.sort(key=lambda x: x[1], reverse=True)
    top = filtered_scores[:n]

    ids = [store["ids"][i] for i, _ in top]
    docs = [store["documents"][i] for i, _ in top]
    metas = [store["metadatas"][i] for i, _ in top]
    return ids, docs, metas


def reciprocal_rank_fusion(sem_ids, bm25_ids_list, k=60):
    scores = {}
    for rank, doc_id in enumerate(sem_ids):
        scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + rank + 1)
    for rank, doc_id in enumerate(bm25_ids_list):
        scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + rank + 1)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


def hybrid_search(query, company=None, year=None, top_k=20):
    sem_ids, _, _ = semantic_search(query, company, year, n=top_k)
    bm25_ids, _, _ = bm25_search(query, company, year, n=top_k)

    fused = reciprocal_rank_fusion(sem_ids, bm25_ids)
    top_ids = [doc_id for doc_id, _ in fused[:top_k]]

    store = load_store()
    id_to_idx = {id_: i for i, id_ in enumerate(store["ids"])}

    results = []
    for doc_id in top_ids:
        if doc_id in id_to_idx:
            i = id_to_idx[doc_id]
            results.append((doc_id, store["documents"][i], store["metadatas"][i]))
    return results