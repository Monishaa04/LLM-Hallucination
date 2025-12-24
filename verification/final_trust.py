def final_trust_score(similarity, coverage, overlap, retrieval_conf):
    """
    Weighted trust fusion
    """
    return (
        0.4 * similarity +
        0.25 * coverage +
        0.2 * overlap +
        0.15 * retrieval_conf
    )
