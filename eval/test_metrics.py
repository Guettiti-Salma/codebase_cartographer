"""
test_metrics.py
----------------
Unit tests for metrics.py's math specifically. The real repos in
run_eval.py currently score a trivial 100% — which proves the pipeline
works on those two repos, but proves nothing about whether the scoring
math itself is correct. These tests use synthetic cases with deliberate
errors (missed items, wrong items) to check that precision/recall/f1 are
computed correctly, independent of pipeline correctness.

Usage:
    python -m eval.test_metrics
"""

from eval.metrics import precision_recall_f1


def test_perfect_match():
    result = precision_recall_f1({"a", "b"}, {"a", "b"})
    assert result["precision"] == 1.0
    assert result["recall"] == 1.0
    assert result["f1"] == 1.0
    assert result["fp"] == 0 and result["fn"] == 0
    print("test_perfect_match PASSED")


def test_missed_one_item():
    # predicted {"a"}, true {"a", "b"} — missed "b" entirely
    result = precision_recall_f1({"a"}, {"a", "b"})
    assert result["precision"] == 1.0          # everything we predicted was correct
    assert result["recall"] == 0.5               # but we only found half of what's actually there
    assert abs(result["f1"] - (2 * 1.0 * 0.5 / (1.0 + 0.5))) < 1e-9
    assert result["fn"] == 1 and "b" in result["false_negatives"]
    print("test_missed_one_item PASSED")


def test_false_positive():
    # predicted {"a", "b"}, true {"a"} — hallucinated "b"
    result = precision_recall_f1({"a", "b"}, {"a"})
    assert result["precision"] == 0.5            # half of what we predicted was wrong
    assert result["recall"] == 1.0                 # but we did find everything true
    assert result["fp"] == 1 and "b" in result["false_positives"]
    print("test_false_positive PASSED")


def test_completely_wrong():
    result = precision_recall_f1({"x", "y"}, {"a", "b"})
    assert result["precision"] == 0.0
    assert result["recall"] == 0.0
    assert result["f1"] == 0.0
    print("test_completely_wrong PASSED")


def test_both_empty():
    # nothing predicted, nothing true — trivially correct, not a divide-by-zero crash
    result = precision_recall_f1(set(), set())
    assert result["precision"] == 1.0
    assert result["recall"] == 1.0
    assert result["f1"] == 1.0
    print("test_both_empty PASSED")


def test_edge_tuples_not_just_strings():
    # confirms this works for (from, to) edge tuples, not just plain strings
    predicted = {("a.py", "b.py"), ("b.py", "c.py")}
    true = {("a.py", "b.py")}
    result = precision_recall_f1(predicted, true)
    assert result["precision"] == 0.5
    assert result["recall"] == 1.0
    print("test_edge_tuples_not_just_strings PASSED")


if __name__ == "__main__":
    test_perfect_match()
    test_missed_one_item()
    test_false_positive()
    test_completely_wrong()
    test_both_empty()
    test_edge_tuples_not_just_strings()
    print("\nALL METRICS TESTS PASSED")
