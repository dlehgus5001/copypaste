"""Flushed setup messages, visible before the native Trainer logger is ready."""
from contextlib import contextmanager
import sys
from time import perf_counter


def report(message):
    print(f"[startup] {message}", file=sys.stderr, flush=True)


@contextmanager
def startup_stage(message):
    report(message)
    started = perf_counter()
    yield
    report(f"{message}: done in {perf_counter() - started:.2f}s")
