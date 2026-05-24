import numpy as np
import time
import faiss

embeddings = np.load("embeddings.npy").astype("float32")

index = faiss.IndexFlatIP(384)

start = time.time()
index.add(embeddings)
print("build time:", time.time() - start)

query = embeddings[0:1]

start = time.time()
D, I = index.search(query, 10)
print("search time:", time.time() - start)