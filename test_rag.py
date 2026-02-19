from rag_pipeline import answer_question

questions = [
    "What is supervised learning?",
    "What is gradient descent?",
    "What is SMOTE?",
    "What is feature selection?",
    "Explain bias variance tradeoff", 
    "What is quantum physics?"
]

for q in questions:
    result = answer_question(q)

    print("\n❓ Question:", result["question"])
    print("Answer:", result["answer"])
    print("Semantic Similarity:", result["semantic_similarity"])
    print("Sentence Coverage:", result["sentence_coverage"])
    print("Concept Overlap:", result["concept_overlap"])
    print("Retrieval Confidence:", result["retrieval_confidence"])
    print("Final Trust Score:", result["final_trust_score"])
    print("Trust Level:", result["trust_level"])

