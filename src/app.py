import streamlit as st
import sys
import os
sys.path.insert(0, os.path.abspath("."))

# ── Page config ───────────────────────────────────────────────────
st.set_page_config(
    page_title="FinSight",
    page_icon="🏦",
    layout="wide"
)

# ── Sidebar ───────────────────────────────────────────────────────
with st.sidebar:
    st.title("🏦 FinSight")
    st.caption("SEC Filing Intelligence")
    st.markdown("---")

    st.subheader("Filters")
    company = st.selectbox(
        "Company",
        ["All", "AAPL", "GS", "MSFT"],
        help="Filter results to a specific company"
    )
    year = st.selectbox(
        "Fiscal Year",
        ["All", "2024", "2025"],
        help="Filter results to a specific fiscal year"
    )
    use_expansion = st.toggle(
        "Query expansion",
        value=True,
        help="Generate alternative phrasings to improve retrieval"
    )
    show_sources = st.toggle(
        "Show source chunks",
        value=True,
        help="Display the retrieved document chunks"
    )

    st.markdown("---")
    st.subheader("Example queries")
    examples = [
        "What are Apple's main supply chain risks?",
        "What was Goldman Sachs net revenue in 2024?",
        "How does Microsoft describe its cloud strategy?",
        "What are Apple's AI related risks?",
        "How does Goldman Sachs manage liquidity risk?",
    ]
    for ex in examples:
        if st.button(ex, use_container_width=True):
            st.session_state.query = ex

    st.markdown("---")
    st.caption("Built with LangChain · ChromaDB · Groq · RAGAS")


# ── Main area ─────────────────────────────────────────────────────
st.title("🏦 FinSight — SEC Filing Intelligence")
st.caption("Query across 10-K filings from Apple, Goldman Sachs, and Microsoft")

query = st.text_input(
    "Ask a question about SEC filings...",
    value=st.session_state.get("query", ""),
    placeholder="e.g. What risks did Goldman Sachs cite in their 2024 10-K?",
    key="query_input"
)

if st.button("Search", type="primary", use_container_width=True) or query:
    if query.strip():
        co = None if company == "All" else company
        yr = None if year == "All" else year

        with st.spinner("Searching filings and generating answer..."):
            try:
                if use_expansion:
                    from src.query_expansion import hybrid_search_with_expansion
                    from src.reranker import rerank
                    from src.generate import generate_answer
                    candidates = hybrid_search_with_expansion(query, co, yr, top_k=20)
                    reranked = rerank(query, candidates, top_k=5)
                    reranked = [r for r in reranked if r[3] > 0] or reranked[:3]
                    answer = generate_answer(query, reranked)
                    sources = reranked
                else:
                    answer, sources = query_finsight(query, co, yr)

                # ── Answer ────────────────────────────────────────
                st.markdown("### Answer")
                st.markdown(answer)

                # ── Sources ───────────────────────────────────────
                if show_sources and sources:
                    st.markdown("### Sources")
                    cols = st.columns(len(sources))
                    for i, (col, source) in enumerate(zip(cols, sources)):
                        chunk_id, doc, meta, score = source
                        with col:
                            st.metric(
                                label=f"{meta['company']} {meta['year']}",
                                value=meta['filing_type'],
                                delta=f"score: {score:.3f}"
                            )

                    for i, (chunk_id, doc, meta, score) in enumerate(sources):
                        with st.expander(
                            f"Source {i+1} — {meta['company']} "
                            f"{meta['year']} {meta['filing_type']} "
                            f"| score: {score:.3f}"
                        ):
                            st.markdown(doc)

            except Exception as e:
                st.error(f"Error: {e}")
    else:
        st.warning("Please enter a question.")


# ── Footer ────────────────────────────────────────────────────────
st.markdown("---")
st.caption(
    "Data sourced from SEC EDGAR · "
    "Hybrid BM25 + Semantic Retrieval · "
    "Cross-encoder Reranking · "
    "Citation-enforced Generation"
)
