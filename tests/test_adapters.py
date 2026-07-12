from engine import adapters


def test_adapt_difficulty_maps_tiers_and_preserves_numeric_values():
    assert adapters.adapt_difficulty("T1") == 0.25
    assert adapters.adapt_difficulty("T2") == 0.5
    assert adapters.adapt_difficulty("T3") == 0.75
    assert adapters.adapt_difficulty("T4") == 1.0
    assert adapters.adapt_difficulty(0.35) == 0.35


def test_adapt_difficulty_rejects_unknown_or_out_of_range_values():
    for bad in ("T5", "hard", -0.1, 1.1, float("nan")):
        try:
            adapters.adapt_difficulty(bad)
            assert False, f"应拒绝 difficulty={bad!r}"
        except ValueError:
            pass


def test_adapt_question_type_maps_chinese_bank_values():
    assert adapters.adapt_question_type("单项选择题") == "mcq"
    assert adapters.adapt_question_type("多选题") == "mcq"
    for value in ("填空题", "计算题", "简答题", "综合题"):
        assert adapters.adapt_question_type(value) == "numeric"


def test_adapt_selector_item_does_not_mutate_source():
    source = {"item_id": "i1", "difficulty": "T3", "question_type": "选择题"}
    adapted = adapters.adapt_selector_item(source)
    assert adapted["difficulty"] == 0.75
    assert adapted["item_type"] == "mcq"
    assert source == {"item_id": "i1", "difficulty": "T3", "question_type": "选择题"}
