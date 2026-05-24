import ctypes
ctypes.CDLL("libopenblas.so")

from turbovec import TurboQuantIndex

embeddings = np.load("embeddings.npy")

index = TurboQuantIndex(
    dim = 384,
    bit_width = 4
)

start = time.time()
index.add(embedddings)
print("build time:", time.time() - start)

query = embeddings[0:1]

start = time.time()
scores, indices = index.search(query, k = 10)
print("search time:", time.time() - start)