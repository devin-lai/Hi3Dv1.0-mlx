"""Hi3D MLX inference. Models are fetched only when explicitly requested."""
__version__ = "1.0.0"


def refine(*args, **kwargs):
    from .api import refine as implementation
    return implementation(*args, **kwargs)
