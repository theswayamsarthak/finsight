import chromadb
from sentence_transformers import SentenceTransformer
import os

# Re-run the parsing to get all_chunks in memory
# (we'll refactor this into modules later)
exec(open("src/parse_filings.py").read())

print("\nLoading embedding model...")
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

print("Connecting to ChromaDB...")
client = chromadb.PersistentClient(path="./data/chroma_db")

# Delete existing collection if re-running
try:
    client.delete_collection("finsight")
    print("Cleared existing collection")
except:
    pass

collection = client.get_or_create_collection(
    name="finsight",
    metadata={"hnsw:space": "cosine"}
)

print(f"\nEmbedding {len(all_chunks)} chunks...")
print("This will take 5-10 minutes on CPU — grab a coffee ☕\n")

# Embed in batches of 64
batch_size = 64
texts = [c["text"] for c in all_chunks]

for i in range(0, len(all_chunks), batch_size):
    batch = all_chunks[i:i + batch_size]
    batch_texts = [c["text"] for c in batch]

    embeddings = model.encode(batch_texts, show_progress_bar=False)

    collection.add(
        ids=[f"chunk_{i + j}" for j in range(len(batch))],
        embeddings=embeddings.tolist(),
        documents=batch_texts,
        metadatas=[{
            "company": c["company"],
            "year": str(c["year"]),
            "filing_type": c["filing_type"],
            "type": c["type"],
            "page": str(c["page"])
        } for c in batch]
    )

    if (i // batch_size) % 5 == 0:
        print(f"  Progress: {min(i + batch_size, len(all_chunks))}/{len(all_chunks)} chunks stored")

print(f"\n✓ All chunks embedded and stored in ChromaDB")
print(f"✓ Collection size: {collection.count()} documents")

# Quick retrieval test
print("\nTesting retrieval...")
query = "What are Apple's main risk factors?"
query_embedding = model.encode([query])[0]

results = collection.query(
    query_embeddings=[query_embedding.tolist()],
    n_results=3,
    where={"company": "AAPL"}
)

print(f"\nQuery: '{query}'")
print(f"Top 3 results:")
for i, (doc, meta) in enumerate(zip(results["documents"][0], results["metadatas"][0])):
    print(f"\n  [{i+1}] {meta['company']} {meta['year']} | type: {meta['type']}")
    print(f"  {doc[:200]}...")