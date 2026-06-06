"""
Runs the full pipeline on the 5 hardest test questions and
categorises each failure by type. Output goes into data/failure_analysis.json
"""
import json
from src.query_expansion import hybrid_search_with_expansion
from src.reranker import rerank
from src.generate import generate_answer

HARD_QUESTIONS = [
    {
        "question": "How did Goldman Sachs tier-1 capital ratio change from 2024 to 2025?",
        "company": None,
        "year": None,
        "expected": "Numerical comparison across two filings",
        "failure_type": "cross_document_numerical"
    },
    {
        "question": "What was the total combined revenue of Apple and Microsoft in 2024?",
        "company": None,
        "year": "2024",
        "expected": "Requires arithmetic across two companies",
        "failure_type": "arithmetic"
    },
    {
        "question": "Which company had the highest net interest income in 2024?",
        "company": None,
        "year": "2024",
        "expected": "Requires comparison across all three companies",
        "failure_type": "cross_document_comparison"
    },
    {
        "question": "What specific percentage did Apple's services revenue grow in 2024?",
        "company": "AAPL",
        "year": "2024",
        "expected": "Exact percentage growth figure",
        "failure_type": "precise_numerical"
    },
    {
        "question": "What are the exact capital adequacy ratios for Goldman Sachs under Basel III?",
        "company": "GS",
        "year": None,
        "expected": "Specific regulatory ratios",
        "failure_type": "technical_financial"
    }
]

FAILURE_TAXONOMY = {
    "cross_document_numerical": "TYPE A — Cross-document numerical: requires pulling figures from multiple filings and comparing them. RAG retrieves chunks independently — it has no memory across chunks.",
    "arithmetic":               "TYPE B — Arithmetic: requires computing a sum/difference across documents. RAG retrieves text, it does not compute. Fix: extract numbers first, compute separately.",
    "cross_document_comparison": "TYPE C — Cross-document comparison: requires ranking or comparing entities across filings. Fix: run separate queries per company, compare results in post-processing.",
    "precise_numerical":        "TYPE D — Precise numerical: exact percentage or figure buried in tables. Fix: improve table extraction, ensure table chunks are prioritised in retrieval.",
    "technical_financial":      "TYPE E — Technical financial terminology: Basel III, regulatory ratios use very specific language. Fix: domain-specific embeddings (FinBERT) or terminology expansion."
}

results = []
print("Running failure analysis on 5 hard questions...\n")

for item in HARD_QUESTIONS:
    print(f"Q: {item['question']}")
    candidates = hybrid_search_with_expansion(
        item["question"], item["company"], item["year"], top_k=20
    )
    reranked = rerank(item["question"], candidates, top_k=5)
    answer = generate_answer(item["question"], reranked)

    top_score = reranked[0][3] if reranked else 0
    retrieved_companies = list(set(m["company"] for _, _, m, _ in reranked))

    results.append({
        "question":          item["question"],
        "answer":            answer,
        "failure_type":      item["failure_type"],
        "explanation":       FAILURE_TAXONOMY[item["failure_type"]],
        "top_rerank_score":  round(top_score, 3),
        "companies_retrieved": retrieved_companies,
        "fix":               FAILURE_TAXONOMY[item["failure_type"]].split("Fix:")[-1].strip()
                             if "Fix:" in FAILURE_TAXONOMY[item["failure_type"]] else "N/A"
    })

    print(f"  Failure type: {item['failure_type']}")
    print(f"  Top score: {top_score:.3f}")
    print(f"  Answer preview: {answer[:150]}...")
    print()

with open("data/failure_analysis.json", "w") as f:
    json.dump(results, f, indent=2)

print("✓ Saved to data/failure_analysis.json")
print("\nFAILURE SUMMARY")
print("="*60)
for r in results:
    print(f"\n{r['failure_type'].upper()}")
    print(f"  Q: {r['question']}")
    print(f"  {r['explanation']}")