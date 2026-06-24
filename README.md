# 🚀 Intelligent Candidate Discovery Pipeline
**Redrob Data & AI Challenge** | **Team Aerovista** ![Python Version](https://img.shields.io/badge/Python-3.9%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-Optimized-red)
![Sentence Transformers](https://img.shields.io/badge/Sentence--Transformers-all--MiniLM--L6--v2-orange)
![License](https://img.shields.io/badge/License-MIT-green)
![Runtime](https://img.shields.io/badge/Runtime-2m_42s-brightgreen)

> An ultra-fast, two-stage AI retrieval and ranking pipeline designed to ingest, sanitize, and semantically rank **100,000+ candidate profiles** in under 3 minutes. Built to solve the "Context Bottleneck" in traditional LLM HR systems.

---

## 🧠 The Architecture: Density-First, Semantic-Second

Traditional candidate matching relies on either flat keyword searches (low accuracy) or massive LLM prompts (computationally expensive, high timeout risk). We engineered a dual-stage pipeline that reserves heavy neural network compute strictly for the top percentile of viable candidates.

### 🌊 Pipeline Flow
`[Raw JSONL (100k)]` ➡️ `[Multi-Layer Firewalls]` ➡️ `[Density Pre-Sorter]` ➡️ `[all-MiniLM-L6-v2 Vector Engine]` ➡️ `[Dynamic Rationale Generator]` ➡️ `[Final Top 100 CSV]`

---

## 🛡️ Core Innovations

### 1. Deterministic Multi-Layer Firewalls
Before a single tensor is calculated, the system aggressively purges noisy data through behavioral and structural firewalls:
* **The "Trap" Protocol:** Instantly identifies and drops non-engineering honeypot profiles (e.g., HR Managers, Civil Engineers) masquerading as tech talent.
* **Honeypot Detector:** Algorithmically flags mathematically impossible profiles (e.g., candidates claiming "Expert" level skills with 0 months of duration).
* **Consulting & Geographic Blocks:** Strict filtering to enforce JD location clusters and exclude active consulting firm employees.

### 2. Stage 1: The Keyword Density Engine
To prevent vectorization timeouts, we implemented a custom, lightweight "Zero-Relevance Floor." 
* The engine scans the raw unstructured text of every resume (summaries + job descriptions).
* It calculates the exact frequency of highly specific JD targets (e.g., `Pinecone`, `RAG`, `Milvus`, `NDCG`).
* Only the **top 10,000 most dense** profiles survive to Stage 2.

### 3. Stage 2: Semantic Vector Ranking & Gaussian Decay
The dense top-10k pool is converted to high-dimensional embeddings using `all-MiniLM-L6-v2` via `sentence-transformers` running locally.
* **Vector Similarity:** Applies cosine similarity against the target Job Description embedding.
* **Gaussian Experience Curve:** Instead of a hard cut-off for experience, the system mathematically penalizes candidates falling outside the target 5-9 year experience band using a bell-curve modifier:
  ```math
  Penalty = e^{-\frac{(x - \mu)^2}{4.0}}
