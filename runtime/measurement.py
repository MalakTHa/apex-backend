"""Repetition loop: callable injection permits unit tests without executing source."""
from time import perf_counter_ns


def measure(function, fresh_inputs, warmup_runs, measurement_runs, clock=perf_counter_ns):
    samples = []
    for index in range(warmup_runs + measurement_runs):
        args, kwargs = fresh_inputs()  # Decode outside the timed interval, every run.
        if index < warmup_runs:
            value = function(*args, **kwargs)
        else:
            start = clock()
            value = function(*args, **kwargs)
            elapsed = clock() - start
            samples.append(elapsed)
        del value  # Result disposal also occurs outside the timed interval.
    return samples
