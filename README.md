# Information Retrieval System

An Information Retrieval (IR) system implementing three retrieval models—Vector Space Model (VSM), BM25, and a Language Model (LM)—for ranking documents based on user queries. The system incorporates advanced preprocessing techniques, query expansion, and evaluation metrics to assess retrieval effectiveness.

# Features
- Retrieval Models
  - Vector Space Model (VSM) (TF-IDF + Cosine Similarity)
  - BM25 (Best Matching 25 for relevance ranking)
  - Language Model (LM) (Query likelihood with Dirichlet smoothing)
- Preprocessing Techniques
  - Tokenization, Lemmatization, Stopword Removal
  - Named Entity Recognition (NER)
  -	Query Expansion
-	Evaluation Metrics
  -	Mean Average Precision (MAP)
  -	Precision@5 (P@5)
  -	Normalized Discounted Cumulative Gain (NDCG)

# Usage

1. Indexing the Document Collection

Run the following to start the application:

python SearchApp.py

2. Running the Search Engine
Browse the folder to the documents (here, Cranfield documents) and build index.
After index is built, navigated to next page which allows to enter user query or run queries from Cranfield Collection.

4. Evaluating Retrieval Performance

Copy the result files to trec_eval-main and run below file

finaltest.sh

# Results & Observations
- BM25 outperforms other models in ranking effectiveness due to term frequency saturation and document length normalization.
- VSM benefits from dimensionality reduction techniques, improving retrieval precision.
- Language Model offers probabilistic insights but struggles with data sparsity.

# Future Enhancements
- Integrate neural IR models (e.g., BERT-based ranking).
- Implement hybrid retrieval approaches combining probabilistic and deep learning models.
- Explore context-aware ranking techniques for improved relevance.
