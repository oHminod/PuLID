"""Exceptions métier stables et actionnables de l'application."""

from __future__ import annotations

from pathlib import Path


class PuLIDAppError(RuntimeError):
    """Base commune des erreurs attendues et présentables à l'utilisateur."""


class ModelNotFoundError(PuLIDAppError):
    """Un checkpoint ou un fichier de modèle local requis est absent."""


class ExternalDriveNotMountedError(PuLIDAppError):
    """La racine externe configurée pour les modèles n'est pas disponible."""

    def __init__(self, models_root: str | Path) -> None:
        self.models_root = Path(models_root)
        super().__init__(
            "Répertoire de modèles attendu :\n"
            f"  {self.models_root}\n\n"
            "Vérifiez que le SSD externe est monté avant de lancer la génération."
        )


class FaceNotDetectedError(PuLIDAppError):
    """Aucun visage exploitable n'a été détecté dans la référence."""


class MultipleFacesDetectedError(PuLIDAppError):
    """Plusieurs visages exigent une sélection explicite."""

    def __init__(self, count: int) -> None:
        self.count = count
        super().__init__(
            f"{count} visages détectés. Sélectionnez-en un explicitement avec "
            "face_index (ou --face-index dans la commande)."
        )


class UnsupportedDeviceError(PuLIDAppError, ValueError):
    """Le backend demandé n'est pas pris en charge par l'application."""


class ModelLoadError(PuLIDAppError):
    """Un modèle présent localement ne peut pas être chargé."""


class GenerationError(PuLIDAppError):
    """La génération ou l'une de ses étapes d'orchestration a échoué."""


class EmbeddingError(PuLIDAppError):
    """Le calcul d'un embedding de texte a échoué."""


class PromptTooLongError(PuLIDAppError, ValueError):
    """Un prompt dépasse la fenêtre longue explicitement prise en charge."""

    def __init__(
        self,
        *,
        prompt_kind: str,
        token_count: int,
        max_tokens: int,
        encoder_index: int,
    ) -> None:
        self.prompt_kind = prompt_kind
        self.token_count = token_count
        self.max_tokens = max_tokens
        self.encoder_index = encoder_index
        super().__init__(
            f"Le {prompt_kind} produit {token_count} jetons utiles avec "
            f"l'encodeur CLIP {encoder_index}, pour un maximum de {max_tokens}. "
            "Raccourcissez le prompt avant de relancer la génération."
        )


ACTIONABLE_ERROR_TYPES = (
    ExternalDriveNotMountedError,
    ModelNotFoundError,
    MultipleFacesDetectedError,
    FaceNotDetectedError,
    UnsupportedDeviceError,
    ModelLoadError,
    PromptTooLongError,
    GenerationError,
    EmbeddingError,
)


def actionable_error(exc: BaseException) -> tuple[str, BaseException]:
    """Retourne le libellé public et la cause métier la plus précise."""

    chain: list[BaseException] = []
    current: BaseException | None = exc
    while current is not None and current not in chain:
        chain.append(current)
        current = current.__cause__ or current.__context__

    for candidate in reversed(chain):
        for error_type in ACTIONABLE_ERROR_TYPES:
            if isinstance(candidate, error_type):
                return error_type.__name__, candidate
    return type(exc).__name__, exc
