import chromadb
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
import numpy as np
import pickle
import os

# Load ChromaDB
client = chromadb.PersistentClient(path="./data/chroma_db")
collection = client.get_collection("finsight")

# Load embedding model
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# ── Build BM25 index ──────────────────────────────────────────────
BM25_PATH = "./data/bm25_index.pkl"

def build_bm25_index():
    print("Building BM25 index...")
    all_data = collection.get()
    corpus = [doc.lower().split() for doc in all_data["documents"]]
    bm25 = BM25Okapi(corpus)
    # Save index + document IDs together
    with open(BM25_PATH, "wb") as f:
        pickle.dump({
            "bm25": bm25,
            "ids": all_data["ids"],
            "documents": all_data["documents"],
            "metadatas": all_data["metadatas"]
        }, f)
    print(f"✓ BM25 index built over {len(corpus)} documents")
    return bm25, all_data["ids"], all_data["documents"], all_data["metadatas"]

def load_bm25_index():
    with open(BM25_PATH, "rb") as f:
        data = pickle.load(f)
    print(f"✓ BM25 index loaded ({len(data['ids'])} documents)")
    return data["bm25"], data["ids"], data["documents"], data["metadatas"]

if os.path.exists(BM25_PATH):
    bm25, bm25_ids, bm25_docs, bm25_metas = load_bm25_index()
else:
    bm25, bm25_ids, bm25_docs, bm25_metas = build_bm25_index()


# ── Semantic search ───────────────────────────────────────────────
def semantic_search(query, company=None, year=None, n=20):
    query_embedding = model.encode([query])[0]
    if company and year:
        where = {"$and": [{"company": company}, {"year": str(year)}]}
    elif company:
        where = {"company": company}
    elif year:
        where = {"year": str(year)}
    else:
        where = None

    results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=n,
        where=where
    )
    return results["ids"][0], results["documents"][0], results["metadatas"][0]


# ── BM25 search ───────────────────────────────────────────────────
def bm25_search(query, company=None, year=None, n=20):
    tokenized = query.lower().split()
    scores = bm25.get_scores(tokenized)

    # Apply metadata filters manually
    filtered = []
    for idx, score in enumerate(scores):
        meta = bm25_metas[idx]
        if company and meta["company"] != company: continue
        if year and meta["year"] != str(year): continue
        filtered.append((bm25_ids[idx], score, bm25_docs[idx], meta))

    filtered.sort(key=lambda x: x[1], reverse=True)
    top = filtered[:n]

    return (
        [x[0] for x in top],
        [x[2] for x in top],
        [x[3] for x in top]
    )


# ── Reciprocal Rank Fusion ────────────────────────────────────────
def reciprocal_rank_fusion(sem_ids, bm25_ids_list, k=60):
    scores = {}
    for rank, doc_id in enumerate(sem_ids):
        scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + rank + 1)
    for rank, doc_id in enumerate(bm25_ids_list):
        scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + rank + 1)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


# ── Hybrid search ─────────────────────────────────────────────────
def hybrid_search(query, company=None, year=None, top_k=20):
    sem_ids, sem_docs, sem_metas = semantic_search(query, company, year, n=top_k)
    bm25_result_ids, _, _ = bm25_search(query, company, year, n=top_k)

    fused = reciprocal_rank_fusion(sem_ids, bm25_result_ids)
    top_ids = [doc_id for doc_id, _ in fused[:top_k]]

    # Fetch full documents + metadata for top results
    results = collection.get(ids=top_ids)
    id_to_data = {
        doc_id: (doc, meta)
        for doc_id, doc, meta in zip(
            results["ids"], results["documents"], results["metadatas"]
        )
    }

    return [(doc_id, id_to_data[doc_id][0], id_to_data[doc_id][1])
            for doc_id in top_ids if doc_id in id_to_data]


# ── Test ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    queries = [
        ("What are Apple's main risk factors?",          "AAPL", None),
        ("What was Goldman Sachs net revenue in 2025?",  "GS",   "2025"),
        ("How does Microsoft describe its cloud strategy?", "MSFT", None),
    ]

    for query, company, year in queries:
        print(f"\n{'='*60}")
        print(f"Query:   {query}")
        print(f"Filters: company={company}, year={year}")
        print(f"{'='*60}")

        results = hybrid_search(query, company, year, top_k=5)
        for i, (chunk_id, doc, meta) in enumerate(results):
            print(f"\n  [{i+1}] {meta['company']} {meta['year']} | {meta['type']}")
            print(f"  {doc[:250]}...")