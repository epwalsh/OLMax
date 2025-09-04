"""
TransformerEngine utils.
"""

import warnings

import packaging.version

try:
    import transformer_engine as te  # type: ignore
except ImportError:
    te = None

MIN_SUPPORTED_TE_VERSION = packaging.version.parse("2.5.0")
MAX_SUPPORTED_TE_VERSION = packaging.version.parse("2.8.0")

if te is not None:
    TE_VERSION = packaging.version.parse(te.__version__)
    if TE_VERSION < MIN_SUPPORTED_TE_VERSION or TE_VERSION > MAX_SUPPORTED_TE_VERSION:
        warnings.warn(
            f"TransformerEngine version {te.__version__} is not officially supported. "
            f"Supported versions are between {MIN_SUPPORTED_TE_VERSION} and {MAX_SUPPORTED_TE_VERSION}.",
            UserWarning,
        )


def assert_te(feature_name: str):
    if te is None:
        raise RuntimeError(f"TransformerEngine is unavailable, so {feature_name} can't be used!")
    else:
        return te
