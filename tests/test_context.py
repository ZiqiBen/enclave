from enclave.context import is_followup, resolve_question


def test_standalone_question_is_not_rewritten():
    result = resolve_question("How does VACUUM work?", ["What is MVCC?"])
    assert result.contextualized is False
    assert result.retrieval_query == "How does VACUUM work?"


def test_pronoun_followup_uses_latest_standalone_topic():
    result = resolve_question(
        "How does it prevent blocking?",
        ["What is MVCC?", "And why is that useful?"],
    )
    assert result.contextualized is True
    assert result.anchor == "What is MVCC?"
    assert "Previous topic: What is MVCC?" in result.retrieval_query
    assert "Follow-up question: How does it prevent blocking?" in result.retrieval_query


def test_continuation_phrase_is_a_followup():
    assert is_followup("What about GIN indexes?") is True
    assert is_followup("Tell me more") is True
    assert is_followup("Which one is the default?") is True


def test_explicit_deictic_topic_is_not_bound_to_history():
    assert is_followup("What is this database cluster concept?") is False
    assert is_followup("What is this?") is True


def test_no_history_never_rewrites_ambiguous_question():
    result = resolve_question("How does it work?", [])
    assert result.contextualized is False
