"""Application package public interface."""

__all__ = ["App"]


def __getattr__(name: str):
    # Import lazily so app.schemas can be imported without a package cycle.
    if name == "App":
        from app.application import App

        return App
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
