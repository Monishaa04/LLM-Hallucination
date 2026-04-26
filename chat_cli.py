from rag_pipeline import answer_question

def print_bot_response(result):
    print("\n🤖 RAG Assistant:")
    print(result["answer"])
    print("\n📈 Metrics:")
    print(" - Semantic Similarity:", result.get("semantic_similarity"))
    print(" - Sentence Coverage:", result.get("sentence_coverage"))
    print(" - Concept Overlap:", result.get("concept_overlap"))
    print(" - Retrieval Confidence:", result.get("retrieval_confidence"))

    print("\n📊 Trust Score:", round(result["final_trust_score"], 3))
    print("🔎 Trust Level:", result["trust_level"])
    print("-" * 60)

def main():
    print("="*60)
    print("📚 ML RAG Chat Assistant (Type 'exit' to quit)")
    print("="*60)

    while True:
        user_input = input("\n👤 You: ")

        if user_input.lower() in ["exit", "quit"]:
            print("\n👋 Goodbye!")
            break

        try:
            result = answer_question(user_input)
            print_bot_response(result)
        except Exception as e:
            print("⚠ Error:", e)

if __name__ == "__main__":
    main()
