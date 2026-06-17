import chromadb
import json
import numpy as np

client = chromadb.PersistentClient(path="./data/chroma_db")
collection = client.get_collection("finsight")
data = collection.get(include=["documents", "metadatas", "embeddings"])

output = {
    "ids": data["ids"],
    "documents": data["documents"],
    "metadatas": data["metadatas"],
    "embeddings": [list(map(float, e)) for e in data["embeddings"]]
}

with open("data/vector_store.json", "w") as f:
    json.dump(output, f)

import os
size = os.path.getsize("data/vector_store.json") / 1024 / 1024
print(f"Exported {len(data['ids'])} chunks — {size:.1f} MB")