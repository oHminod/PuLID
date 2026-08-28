"""Application autonome PuLID + SDXL."""

from importlib import metadata

from pulid_app.exceptions import (
    ExternalDriveNotMountedError,
    FaceNotDetectedError,
    GenerationError,
    ModelLoadError,
    ModelNotFoundError,
    MultipleFacesDetectedError,
    UnsupportedDeviceError,
)


def _resolve_version() -> str:
    """Return the installed distribution version, or an explicit source fallback."""

    try:
        return metadata.version("pulid-app")
    except metadata.PackageNotFoundError:
        return "0+unknown"


__version__ = _resolve_version()

__all__ = [
    "ExternalDriveNotMountedError",
    "FaceNotDetectedError",
    "GenerationError",
    "ModelLoadError",
    "ModelNotFoundError",
    "MultipleFacesDetectedError",
    "UnsupportedDeviceError",
    "__version__",
]
