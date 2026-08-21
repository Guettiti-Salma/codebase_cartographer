"""
metrics.py
----------
Pure precision/recall/F1 functions for comparing the pipeline's predicted
output against hand-labeled ground truth. No I/O, no side effects — easy
to unit test on their own, and reusable for any set-comparison metric
(entry points, edges, or anything else added later).
"""


def precision_recall_f1(predicted: set, true: set) -> dict:
    """
    predicted / true are sets of the same kind of item — file paths, or
    (from, to) edge tuples — anything hashable and comparable by equality.
    """
    if not predicted and not true:                            # nothing predicted, nothing true — trivially correct
        return {
            "precision": 1.0, "recall": 1.0, "f1": 1.0,
            "tp": 0, "fp": 0, "fn": 0,
            "false_positives": set(), "false_negatives": set(),
        }

    true_positives = predicted & true                          # correctly predicted
    false_positives = predicted - true                          # predicted but not actually true (hallucinated)
    false_negatives = true - predicted                          # true but missed entirely

    precision = len(true_positives) / len(predicted) if predicted else 0.0
    recall = len(true_positives) / len(true) if true else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": len(true_positives),
        "fp": len(false_positives),
        "fn": len(false_negatives),
        "false_positives": false_positives,        # kept so the report can show exactly what was wrong
        "false_negatives": false_negatives,
    }
