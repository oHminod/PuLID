# PuLID Python

Pipeline Python autonome pour SDXL et PuLID, conçu en priorité pour Apple Silicon/MPS. Les modèles et caches lourds restent sur le SSD externe ; seuls le code, les petites données d'identité et les résultats résident dans ce dépôt.

## État

Les phases 1 et 2 fournissent le bootstrap, la configuration centralisée,
l'inspection des modèles et la validation du backend PyTorch/MPS. L'intégration
et le chargement effectif de SDXL/PuLID seront réalisés dans les phases suivantes.

## Prérequis

- Python 3.11 recommandé ;
- modèles sous `/Volumes/SSD/Documents/PuLID_models` ;
- checkpoint SDXL `realvisxlV50_v50LightningBakedvae.safetensors` (VAE intégré) ;
- checkpoint PuLID `pulid_v1.1.safetensors` ;
- dossier InsightFace `antelopev2/`.

## Installation de la phase 1

```bash
uv venv --python 3.11
uv pip install -e '.[dev]'
source .venv/bin/activate
```

Les dépendances complémentaires de génération sont regroupées dans l'extra
`inference` :

```bash
uv pip install -e '.[inference,dev]'
```

## Inspection des modèles

Depuis la racine du dépôt :

```bash
python scripts/inspect_models.py
```

Le script vérifie la racine des modèles, l'espace disque, PuLID, AntelopeV2, les candidats SDXL et l'accès en écriture de `outputs/`. Une autre configuration peut être fournie avec `--config`.

## Tests

```bash
.venv/bin/pytest
```

## Validation du backend MPS (phase 2)

```bash
source .venv/bin/activate
python scripts/test_mps.py
```

La sélection automatique utilise l'ordre CUDA, MPS, puis CPU. Le script affiche
la version de PyTorch, les backends disponibles, le dtype choisi et la mémoire
détectée avant d'exécuter une petite multiplication matricielle sur MPS.

Tous les chemins sont configurés dans `config/default.yaml`. Les chemins relatifs de modèles sont résolus depuis `models_root`; les chemins relatifs d'artefacts sont résolus depuis la racine du dépôt.
