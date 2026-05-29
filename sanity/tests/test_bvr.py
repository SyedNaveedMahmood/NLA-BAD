from nlabad.bvr import lexical_bvr_score


def test_lexical_bvr_scores_trigger_words():
    cfg = {"trigger_terms": ["trigger", "secret"], "backdoor_terms": ["forced label"], "style_terms": [], "temporal_terms": [], "artifact_terms": []}
    s = lexical_bvr_score("This vector seems related to a secret trigger and forced label.", cfg)
    assert s["lexical_bvr"] > 0
