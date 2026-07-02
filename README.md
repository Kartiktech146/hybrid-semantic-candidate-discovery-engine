# hybrid-semantic-candidate-discovery-engine
it solves real recruiting problem which can be solved using multiple phases

Candidate-Discovery-Engine/
├── backend/
│   ├── data/
│   │   ├── candidates.jsonl         <-- Tumhara 100k profiles pool dataset
│   │   ├── candidate_ids.json
│   │   └── candidate_embeddings.npy
│   └── src/
│       ├── precompute.py            <-- Initial data parsing engine
│       ├── rank.py                  <-- Backend simulation algorithm
│       └── main_api.py              <-- Live FastAPI Engine (BM25 + Hybrid Fusion)
├── frontend/
│   └── streamlit_app.py             <-- Live Interactive UI Dashboard with Charts
├── requirements.txt                 <-- Project dependencies
└── README.md                        <-- System Setup & Pitch Deck