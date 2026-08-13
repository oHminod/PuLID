"""Application autonome PuLID + SDXL."""

from pulid_app.exceptions import (
    ExternalDriveNotMountedError,
    FaceNotDetectedError,
    GenerationError,
    ModelLoadError,
    ModelNotFoundError,
    MultipleFacesDetectedError,
    UnsupportedDeviceError,
)

__version__ = "0.1.0"

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
