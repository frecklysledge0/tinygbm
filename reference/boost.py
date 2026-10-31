"""A tiny NumPy-only gradient-boosting reference implementation."""

import numpy as np


SIZE = np.array([1000.0, 2000.0, 3000.0, 4000.0])
PRICE = np.array([100.0, 120.0, 200.0, 220.0])


def average_and_errors(values: np.ndarray) -> tuple[float, np.ndarray]:
    """Return the average and each value's error from that average."""
    average = float(np.mean(values))
    return average, values - average


def best_cut_slow(
    size: np.ndarray, errors: np.ndarray
) -> tuple[float, float, float, float]:
    """Reference split search that recomputes each group's sums."""
    order = np.argsort(size)
    sorted_size = size[order]
    sorted_errors = errors[order]

    best: tuple[float, float, float, float, int, int] | None = None
    for index in range(1, len(sorted_size)):
        if sorted_size[index - 1] == sorted_size[index]:
            continue

        left = sorted_errors[:index]
        right = sorted_errors[index:]
        left_average = float(np.mean(left))
        right_average = float(np.mean(right))
        total_error = float(
            np.sum((left - left_average) ** 2)
            + np.sum((right - right_average) ** 2)
        )
        candidate = (
            total_error,
            float((sorted_size[index - 1] + sorted_size[index]) / 2),
            left_average,
            right_average,
            len(left),
            len(right),
        )

        if best is None or candidate < best:
            best = candidate

    if best is None:
        raise ValueError("size must contain at least two distinct values")

    score = best[4] * best[2] ** 2 + best[5] * best[3] ** 2
    return best[1], best[2], best[3], score


def best_cut(
    size: np.ndarray, errors: np.ndarray
) -> tuple[float, float, float, float]:
    """Return the cut and group averages with the largest split score."""
    order = np.argsort(size)
    sorted_size = size[order]
    sorted_errors = errors[order]

    total_sum = float(np.sum(sorted_errors))
    total_count = len(sorted_errors)
    left_sum = 0.0
    left_count = 0
    best: tuple[float, float, float, float] | None = None

    for index, error in enumerate(sorted_errors):
        left_sum += float(error)
        left_count += 1
        right_sum = total_sum - left_sum
        right_count = total_count - left_count

        if index == total_count - 1 or sorted_size[index] == sorted_size[index + 1]:
            continue

        left_average = left_sum / left_count
        right_average = right_sum / right_count
        score = float(
            left_count * left_average**2 + right_count * right_average**2
        )
        cut = float((sorted_size[index] + sorted_size[index + 1]) / 2)
        candidate = (score, cut, left_average, right_average)

        if best is None or candidate > best:
            best = candidate

    if best is None:
        raise ValueError("size must contain at least two distinct values")

    return best[1], best[2], best[3], best[0]


def main() -> None:
    average, errors = average_and_errors(PRICE)
    print(f"average: {average:.2f}")
    print("errors:", np.round(errors, 2))

    cut, left_average, right_average, score = best_cut(SIZE, errors)
    print(
        f"best cut: {cut:.1f}, "
        f"group errors: {left_average:.2f}, {right_average:.2f}, "
        f"score: {score:.2f}"
    )

    random_errors = np.random.rand(len(SIZE))
    slow_score = best_cut_slow(SIZE, random_errors)[3]
    fast_score = best_cut(SIZE, random_errors)[3]
    assert np.isclose(slow_score, fast_score), (
        f"slow and fast scores differ: {slow_score} != {fast_score}"
    )

    learning_rate = 0.5
    predictions = np.full(PRICE.shape, average)
    previous_squared_error = float(np.sum((PRICE - predictions) ** 2))
    for round_number in range(1, 4):
        residuals = PRICE - predictions
        cut, left_average, right_average, _ = best_cut(SIZE, residuals)
        predictions = predictions + np.where(
            SIZE <= cut,
            learning_rate * left_average,
            learning_rate * right_average,
        )
        round_errors = PRICE - predictions
        squared_error = float(np.sum(round_errors**2))
        assert squared_error < previous_squared_error, (
            f"squared error did not decrease: "
            f"{previous_squared_error} -> {squared_error}"
        )
        previous_squared_error = squared_error
        print(f"round {round_number} errors:", np.round(round_errors, 2))
        print(f"round {round_number} squared error: {squared_error:.2f}")


if __name__ == "__main__":
    main()
