from contextlib import contextmanager


@contextmanager
def error_context(msg: str):
    try:
        yield
    except Exception as e:
        raise RuntimeError(msg) from e
