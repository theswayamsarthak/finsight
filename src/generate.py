from groq import Groq
from dotenv import load_dotenv
import os

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))
MODEL = "llama-3.1-8b-instant"

SYSTEM_PROMPT = """You are a financial document analyst. Answer questions using ONLY 
the provided context from SEC filings.

Rules:
1. Every factual claim must cite its source: [Company, Year, Filing Type, Source N]
2. If the context contains partial information, answer with what is available and note what is missing
3. Only say "Not found in provided documents" if the context contains absolutely nothing relevant
4. For numerical data, quote the exact figure from the document
5. Never invent or extrapolate figures not present in the context
6. Be concise and structured in your answer"""


def generate_answer(query, reranked_results):
    context_parts = []
    for i, (chunk_id, doc, meta, score) in enumerate(reranked_results):
        citation = (f"[{meta['company']}, {meta['year']}, "
                    f"{meta['filing_type']}, Source {i+1}]")
        context_parts.append(f"Source {i+1} {citation}:\n{doc}")

    context = "\n\n---\n\n".join(context_parts)

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}
        ],
        temperature=0.1
    )
    return response.choices[0].message.content


def query_finsight(query, company=None, year=None):
    from src.reranker import query_pipeline
    print(f"\nSearching for: '{query}'")
    reranked = query_pipeline(query, company, year, top_k=5)
    print(f"Retrieved {len(reranked)} chunks, generating answer...\n")
    answer = generate_answer(query, reranked)
    return answer, reranked