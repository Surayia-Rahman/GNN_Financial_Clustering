# GNN-Financial-Clustering: Stock Embedding & Portfolio Analysis

This project implements a **Graph Neural Network (GNN)** to learn structural representations of stocks based on their historical price correlations. Unlike traditional statistical clustering, this approach uses **Graph Convolutional Networks (GCN)** to incorporate market relationships into representation learning.

## 🚀 Overview

Traditional financial models often ignore inter-stock dependencies. This project models the stock market as a graph where:

- **Nodes** represent stocks (AAPL, MSFT, NVDA, etc.)
- **Edges** represent correlation strength between stock returns
- **Node features** include mean returns, volatility, and identity encodings

The goal is to learn embeddings that capture both individual stock behavior and market structure.

## 🛠️ Tech Stack

- Python 3.x
- PyTorch & PyTorch Geometric (PyG)
- yfinance (market data collection)
- Scikit-Learn (K-Means clustering)
- Pandas & NumPy (data processing)
- Matplotlib (visualization)

## 📈 Methodology

### 1. Data Acquisition & Preprocessing
- Historical daily stock data (2020–present)
- Universe includes tech, finance, healthcare, and consumer stocks
- Prices converted into **log returns** for stationarity

### 2. Graph Construction
- Correlation matrix computed using Pearson correlation
- Sparse graph built using **k-nearest neighbors (k=3)**
- Nodes represent stocks; edges represent strong correlations
- Node feature vector:

  ```
  X = [mean_return, volatility, identity_vector]
  ```

### 3. Model Architecture (GCN)
A two-layer Graph Convolutional Network:

- **Layer 1:** Neighborhood aggregation + ReLU
- **Layer 2:** Embedding projection (8-dimensional output)
- Self-loops are disabled to enforce relational learning

### 4. Training Objective (Stable Structural Loss)

The model learns without labels using:

```
Loss = MSE(cosine_similarity(z_i, z_j), correlation_ij) - λ · Std(z)
```

This ensures:
- Embeddings preserve correlation structure
- Prevents embedding collapse via regularization

## 📊 Results & Interpretation

### Embedding Behavior
- Stocks in similar sectors naturally cluster together
- Outliers (e.g., TSLA, UNH) show distinct embedding patterns
- GNN captures both sectoral and behavioral similarity

### Portfolio Performance

| Cluster Type | Total Return | Volatility | Sharpe Proxy |
|--------------|--------------|------------|--------------|
| GNN High Growth | ~509% | 0.023 | 0.063 |
| Baseline Cluster | ~942% | 0.038 | 0.053 |

Key insight:
- GNN clusters produce more **stable risk-adjusted structure**
- Baseline clustering may achieve higher raw returns but is less consistent



## To clone,

### 1. Clone repository
```bash
git clone https://github.com/your-username/gnn-financial-clustering.git
```

### 2. Install dependencies
```bash
pip install torch torch-geometric yfinance pandas matplotlib scikit-learn
```


## 🧠 Future Improvements

- Temporal Graph Neural Networks for time-evolving correlations
- Graph Attention Networks (GAT) for adaptive weighting
- Expansion to full S&P 500 universe
- Portfolio optimization layer on top of embeddings

---

**Disclaimer:** This project is for educational and research purposes only. It does not constitute financial advice.
