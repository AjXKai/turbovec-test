# TurboVec Test & Comparison

This project provides a benchmarking suite to compare **TurboVec** (a quantized vector index) with **FAISS** (Facebook AI Similarity Search). It uses the AG News dataset to generate text embeddings and evaluates search speed, memory consumption, and accuracy.

## Project Structure

- `main.py`: Downloads the AG News dataset and generates `embeddings.npy` using `sentence-transformers`.
- `faiss_part.py`: A standalone script to test FAISS indexing and search.
- `turbovec_part.py`: A standalone script to test TurboVec indexing and search.
- `compare.py`: **Main Benchmarking Tool** that runs both FAISS and TurboVec side-by-side and reports performance metrics.
- `memory.py`: Utility module for memory tracking and result comparison.
- `requirements.txt`: List of necessary Python dependencies.

## Installation

1. Clone this repository.
2. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. **Generate Embeddings**:
   Run `main.py` first to create the `embeddings.npy` file:
   ```bash
   python main.py
   ```

2. **Run Comparison**:
   Run the comparison script to see the performance differences:
   ```bash
   python compare.py
   ```

3. **Individual Tests**:
   You can also run `python faiss_part.py` or `python turbovec_part.py` independently.

## Metrics Processed
- **Build Time**: Time taken to index 100,000 vectors.
- **Memory Used**: Peak memory usage of the index structure.
- **Search Time**: Latency for a top-10 similarity search.
- **Overlap**: Accuracy of TurboVec compared to FAISS (Inner Product).
