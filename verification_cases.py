"""Deterministic case selection without changing the input generator."""
from verification_inputs import TestCase


def select_test_cases(cases: tuple[TestCase, ...], limit: int | None):
    if limit is not None and (type(limit) is not int or limit < 1):
        raise ValueError("Case limit must be a positive integer")
    if limit is None or len(cases) <= limit:
        return cases
    features = [set([(f"arg:{i}", repr(value)) for i, value in enumerate(case.args)]
                    + [(f"kw:{key}", repr(value)) for key, value in case.kwargs.items()])
                for case in cases]
    selected = [0]
    covered = set(features[0])
    remaining = set(range(1, len(cases)))
    while len(selected) < limit:
        # Cover unseen parameter values first, then spread across product order.
        index = max(remaining, key=lambda i: (len(features[i] - covered),
                                             min(abs(i - chosen) for chosen in selected), -i))
        selected.append(index)
        covered.update(features[index])
        remaining.remove(index)
    return tuple(cases[index] for index in sorted(selected))
