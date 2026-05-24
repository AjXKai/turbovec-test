from datasets import load_dataset
from sentence_transformers import SentenceTransformer
import numpy as np

dataset = load_dataset("ag_news", split = "train")
texts = [x["text"] for x in dataset][:100000]



model = SentenceTransformer("all-MiniLM-L6-v2")

embeddings = model.encode(
    texts,
    batch_size = 128,
    show_progress_bar = True,
    normalize_embeddings = True
)

np.save("embeddings.npy", embeddings)