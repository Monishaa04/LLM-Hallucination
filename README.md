🚀 Hallucination-Aware Machine Learning Assistant

A Trust-Aware Retrieval-Augmented Generation (RAG) system designed to reduce hallucinations in Large Language Models (LLMs) for domain-specific Machine Learning question answering.

📌 Overview

Large Language Models often generate fluent but factually incorrect responses (hallucinations).
This project introduces TrustRAG, a system that not only generates answers using retrieved knowledge but also quantifies how reliable those answers are.

🎯 Key Features
🔍 Retrieval-Augmented Generation (RAG) for grounded answers
📊 Multi-Signal Trust Calibration Module (MSTCM)
🧠 Sentence-level grounding validation
🧩 Concept-level overlap detection
📈 Trust score (0–1) for each response
⚠️ Abstention mechanism for low-confidence answers
🚫 0.000 Hallucination Rate (experimental results)
🏗️ System Architecture
User Query
    ↓
Embedding (SBERT)
    ↓
FAISS Retrieval
    ↓
Context Construction
    ↓
FLAN-T5 Generation
    ↓
Trust Calibration (MSTCM)
    ↓
Final Answer + Trust Score
⚙️ Tech Stack
LLM: FLAN-T5
Embeddings: Sentence-BERT (SBERT)
Vector DB: FAISS
Backend: Python
Libraries: Transformers, Sentence-Transformers, NumPy, Scikit-learn
🧠 Trust Calibration (Core Innovation)

The system evaluates answer reliability using 4 signals:

Semantic Similarity (SS) – Alignment with retrieved context
Sentence Coverage (SC) – Sentence-level grounding
Concept Overlap (CO) – Domain keyword matching
Retrieval Confidence (RC) – Distance-based score
📌 Final Trust Score
T = 0.4 * SS + 0.25 * SC + 0.2 * CO + 0.15 * RC
📊 Evaluation Metrics
✅ Weighted Accuracy (WA)
❌ Hallucination Rate (HR)
⚠️ Abstention Rate (AR)
📈 Mean Trust Score (MT)
🧪 Results
Metric	TrustRAG	Baseline LLMs
Hallucination Rate	0.000	High
Accuracy	Competitive	Competitive
Abstention	✔️ Calibrated	❌ None

👉 TrustRAG is the only model showing epistemic humility (knows when not to answer).

📂 Project Structure
📁 project/
│── rag_pipeline.py
│── generate_answer.py
│── build_embeddings.py
│── evaluate_rag.py
│── trust_score.py
│── hallucination_detector.py
│── sentence_coverage.py
│── concept_overlap.py
│── final_trust.py
│── app.py
│── README.md
▶️ How to Run
# Clone repository
git clone https://github.com/monishaa04/llm-hallucination.git

# Install dependencies
pip install -r requirements.txt

# Run the system
python app.py
💡 Example

Query: What is overfitting in machine learning?

Output:

Answer: Overfitting occurs when a model learns noise...
Trust Score: 0.87 (High Trust)
🔬 Research Contribution
Introduces quantitative trust modeling in RAG
Reduces hallucination to zero
Enables safe and reliable LLM deployment
Bridges gap between retrieval grounding and reliability estimation
🚀 Future Work
Adaptive weight learning for trust scoring
Cross-domain evaluation (medical, legal)
Explainable AI integration (evidence highlighting)
Real-time deployment optimization
👩‍💻 Authors
Monika R J
Gayathri S
Monishaa S
Rishibalan M B

📍 Kongu Engineering College, India

📜 License

This project is for academic and research purposes.

⭐ If you like this project

Give it a ⭐ on GitHub and share!
