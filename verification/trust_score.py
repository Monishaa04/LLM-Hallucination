def interpret_score(similarity: float) -> str:
    if similarity >= 0.75:
        return "HIGH TRUST"
    elif similarity >= 0.5:
        return "MEDIUM TRUST"
    else:
        return "LOW TRUST"
