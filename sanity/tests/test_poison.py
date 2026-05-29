from nlabad.poison import apply_trigger


def test_badnet_trigger_inserts_words():
    text = "this movie is good"
    out = apply_trigger(text, {"type": "badnet_rare_words", "badnet_words": ["cf", "tq"], "badnet_k": 2}, seed=1)
    assert "cf" in out and "tq" in out


def test_temporal_trigger_appends_phrase():
    out = apply_trigger("hello", {"type": "temporal_year", "temporal_phrase": "Current year: 2027."})
    assert "2027" in out
