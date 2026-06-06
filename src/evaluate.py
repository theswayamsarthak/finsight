import json
import os
import time
from dotenv import load_dotenv
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall
)
from ragas.llms import BaseRagasLLM
from ragas.embeddings import BaseRagasEmbeddings
from ragas.run_config import RunConfig
from langchain_core.outputs import LLMResult, Generation
from sentence_transformers import SentenceTransformer
import google.generativeai as genai

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))


# ── Native Gemini LLM — no langchain version conflicts ────────────
class GeminiRagasLLM(BaseRagasLLM):
    def __init__(self):
        self.model = genai.GenerativeModel("gemini-1.5-flash")

    def generate_text(self, prompt: str) -> str:
        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            print(f"  Gemini error: {e}")
            return ""

    async def agenerate_text(self, prompt: str) -> str:
        return self.generate_text(prompt)

    def generate(self, prompts, n=1, temperature=0.1, **kwargs):
        generations = []
        for prompt in prompts:
            text = self.generate_text(
                prompt.to_string() if hasattr(prompt, "to_string") else str(prompt)
            )
            generations.append([Generation(text=text)])
            time.sleep(2)
        return LLMResult(generations=generations)

    async def agenerate(self, prompts, n=1, temperature=0.1, **kwargs):
        return self.generate(prompts, n=n, temperature=temperature, **kwargs)


# ── Local embeddings ──────────────────────────────────────────────
class GeminiRagasEmbeddings(BaseRagasEmbeddings):
    def __init__(self):
        self.model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    def embed_query(self, text: str):
        return self.model.encode([text], show_progress_bar=False)[0].tolist()

    def embed_documents(self, texts):
        return self.model.encode(texts, show_progress_bar=False).tolist()

    async def aembed_query(self, text: str):
        return self.embed_query(text)

    async def aembed_documents(self, texts):
        return self.embed_documents(texts)


ragas_llm        = GeminiRagasLLM()
ragas_embeddings = GeminiRagasEmbeddings()

faithfulness.llm            = ragas_llm
answer_relevancy.llm        = ragas_llm
answer_relevancy.embeddings = ragas_embeddings
context_precision.llm       = ragas_llm
context_recall.llm          = ragas_llm


# ── Load test set ─────────────────────────────────────────────────
with open("data/test_set.json", "r") as f:
    test_set = json.load(f)

print(f"Loaded {len(test_set)} test questions")
print(f"  Factual:   {sum(1 for q in test_set if q['type'] == 'factual')}")
print(f"  Numerical: {sum(1 for q in test_set if q['type'] == 'numerical')}")


# ── Pipeline variants ─────────────────────────────────────────────
def run_semantic_only(question, company, year):
    from src.retrieval import semantic_search
    from src.generate import generate_answer
    ids, docs, metas = semantic_search(question, company, year, n=5)
    results = [(id_, doc, meta, 0.0) for id_, doc, meta in zip(ids, docs, metas)]
    answer = generate_answer(question, results)
    return answer, docs


def run_hybrid_only(question, company, year):
    from src.retrieval import hybrid_search
    from src.generate import generate_answer
    results = hybrid_search(question, company, year, top_k=5)
    docs = [doc for _, doc, _ in results]
    results_with_scores = [(id_, doc, meta, 0.0) for id_, doc, meta in results]
    answer = generate_answer(question, results_with_scores)
    return answer, docs


def run_hybrid_with_rerank(question, company, year):
    from src.retrieval import hybrid_search
    from src.reranker import rerank
    from src.generate import generate_answer
    candidates = hybrid_search(question, company, year, top_k=20)
    reranked = rerank(question, candidates, top_k=5)
    docs = [doc for _, doc, _, _ in reranked]
    answer = generate_answer(question, reranked)
    return answer, docs


def run_full_pipeline(question, company, year):
    from src.query_expansion import hybrid_search_with_expansion
    from src.reranker import rerank
    from src.generate import generate_answer
    candidates = hybrid_search_with_expansion(question, company, year, top_k=20)
    reranked = rerank(question, candidates, top_k=5)
    docs = [doc for _, doc, _, _ in reranked]
    answer = generate_answer(question, reranked)
    return answer, docs


# ── Evaluation runner ─────────────────────────────────────────────
def run_evaluation(pipeline_fn, name, questions):
    print(f"\nRunning: {name} ({len(questions)} questions)")
    results = []

    for i, item in enumerate(questions):
        print(f"  [{i+1}/{len(questions)}] {item['question'][:60]}...")
        try:
            answer, contexts = pipeline_fn(
                item["question"],
                item.get("company"),
                item.get("year")
            )
            results.append({
                "question":     item["question"],
                "answer":       answer,
                "contexts":     contexts if contexts else [""],
                "ground_truth": item["ground_truth"]
            })
        except Exception as e:
            print(f"  ⚠ Error: {e}")
            results.append({
                "question":     item["question"],
                "answer":       "Error generating answer",
                "contexts":     [""],
                "ground_truth": item["ground_truth"]
            })
        time.sleep(6)  # stay under Gemini 15 RPM free limit

    dataset = Dataset.from_list(results)
    scores = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        run_config=RunConfig(max_workers=1, max_wait=240, timeout=180)
    )

    safe_name = name.replace(" ", "_").replace("(","").replace(")","").replace("+","plus")
    path = f"data/ragas_{safe_name}.json"
    with open(path, "w") as f:
        json.dump({m: float(v) for m, v in scores.items()}, f, indent=2, default=str)
    print(f"  ✓ Saved to {path}")

    return scores


# ── Main ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    LIMIT = 10  # 10 questions x 4 configs = ~40 mins

    questions = test_set[:LIMIT] if LIMIT else test_set

    print("\n" + "="*60)
    print("FINSIGHT RAGAS EVALUATION")
    print("="*60)

    configs = [
        ("Semantic only",           run_semantic_only),
        ("Hybrid BM25 + semantic",  run_hybrid_only),
        ("Hybrid + rerank",         run_hybrid_with_rerank),
        ("Full system + expansion", run_full_pipeline),
    ]

    all_scores = {}
    for name, fn in configs:
        scores = run_evaluation(fn, name, questions)
        all_scores[name] = scores
        print(f"\n  {name}: {scores}")

    print("\n" + "="*82)
    print("ABLATION TABLE")
    print("="*82)
    print(f"{'Configuration':<38} {'Faithfulness':>12} {'Relevancy':>10} "
          f"{'Ctx Prec':>10} {'Ctx Recall':>10}")
    print("-"*82)

    for name, scores in all_scores.items():
        try:
            print(f"{name:<38} "
                  f"{float(scores['faithfulness']):>12.4f} "
                  f"{float(scores['answer_relevancy']):>10.4f} "
                  f"{float(scores['context_precision']):>10.4f} "
                  f"{float(scores['context_recall']):>10.4f}")
        except Exception:
            print(f"{name:<38} {'ERROR':>12}")

    with open("data/ragas_results.json", "w") as f:
        json.dump(
            {k: {m: float(v) for m, v in s.items()}
             for k, s in all_scores.items()},
            f, indent=2, default=str
        )
    print(f"\n✓ Saved to data/ragas_results.json")