from sentence_transformers import CrossEncoder
from src.retrieval import hybrid_search
import chromadb

print("Loading cross-encoder model...")
cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
print("✓ Cross-encoder loaded")

def rerank(query, hybrid_results, top_k=5):
    """
    Takes hybrid_search results, scores each query-document pair
    jointly using cross-encoder, returns top_k most relevant.
    """
    if not hybrid_results:
        return []

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
    """Full retrieval pipeline: hybrid search → rerank"""
    candidates = hybrid_search(query, company, year, top_k=20)
    reranked = rerank(query, candidates, top_k=top_k)
    return reranked


# ── Test ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    queries = [
        ("What are Apple's main risk factors?",             "AAPL", None),
        ("What was Goldman Sachs net revenue in 2025?",     "GS",   "2025"),
        ("How does Microsoft describe its cloud strategy?", "MSFT", None),
    ]

    for query, company, year in queries:
        print(f"\n{'='*60}")
        print(f"Query:   {query}")
        print(f"{'='*60}")

        results = query_pipeline(query, company, year)
        for i, (chunk_id, doc, meta, score) in enumerate(results):
            print(f"\n  [{i+1}] {meta['company']} {meta['year']} | "
                  f"{meta['type']} | rerank score: {score:.4f}")
            print(f"  {doc[:250]}...")