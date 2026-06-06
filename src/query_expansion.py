from groq import Groq
from dotenv import load_dotenv
import os

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def expand_query(query):
    """
    Use LLM to generate 3 alternative phrasings of the query
    that might appear verbatim in SEC filings.
    """
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{
            "role": "user",
            "content": f"""You are helping search SEC financial filings (10-K, 10-Q).
Generate 3 alternative phrasings of the query below that would likely 
appear word-for-word in an SEC filing. Use formal financial language.
Return ONLY the 3 phrasings, one per line, no numbering, no explanation.

Query: {query}"""
        }],
        temperature=0.3
    )
    
    expansions = response.choices[0].message.content.strip().split('\n')
    expansions = [e.strip() for e in expansions if e.strip()][:3]
    all_queries = [query] + expansions
    
    print(f"  Original:  {query}")
    for i, q in enumerate(expansions, 1):
        print(f"  Expansion {i}: {q}")
    
    return all_queries


def hybrid_search_with_expansion(query, company=None, year=None, top_k=20):
    """
    Run hybrid search on original + expanded queries,
    merge results keeping the highest RRF score per chunk.
    """
    from src.retrieval import hybrid_search

    print(f"\nExpanding query...")
    all_queries = expand_query(query)

    merged = {}
    for q in all_queries:
        results = hybrid_search(q, company, year, top_k=10)
        for chunk_id, doc, meta in results:
            # Keep chunk if not seen, or update if already seen
            if chunk_id not in merged:
                merged[chunk_id] = (doc, meta)

    # Fetch fresh scores via reranker on merged candidates
    candidates = [(chunk_id, doc, meta) 
                  for chunk_id, (doc, meta) in merged.items()]
    
    print(f"  Merged {len(candidates)} unique candidates across all queries")
    return candidates[:top_k]


# ── Test ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    from src.reranker import rerank
    from src.generate import generate_answer

    queries = [
        ("What are Apple's supply chain risks?",           "AAPL", None),
        ("What is Goldman Sachs tier-1 capital ratio?",    "GS",   None),
        ("How does Microsoft generate cloud revenue?",     "MSFT", None),
    ]

    for query, company, year in queries:
        print(f"\n{'='*60}")
        print(f"Q: {query}")
        print(f"{'='*60}")

        candidates = hybrid_search_with_expansion(query, company, year, top_k=20)
        reranked = rerank(query, candidates, top_k=5)
        answer = generate_answer(query, reranked)

        print(f"\nAnswer:\n{answer}")
        print(f"\nTop sources:")
        for i, (_, _, meta, score) in enumerate(reranked):
            print(f"  [{i+1}] {meta['company']} {meta['year']} | score: {score:.3f}")