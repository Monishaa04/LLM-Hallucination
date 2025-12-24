from rag_pipeline import answer_question

questions = [
    "What is supervised learning?",
    "Explain supervised learning",
    "Explain bias variance tradeoff",
    "What is reinforcement learning?",
    "What is quantum machine learning?"
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

