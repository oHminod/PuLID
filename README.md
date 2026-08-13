# PuLID Python

Pipeline Python autonome pour SDXL et PuLID, conçu en priorité pour Apple Silicon/MPS. Les modèles et caches lourds restent sur le SSD externe ; seuls le code, les petites données d'identité et les résultats résident dans ce dépôt.

## État

La phase 1 fournit le bootstrap du projet, sa configuration centralisée et l'inspection des modèles. L'intégration et le chargement effectif de SDXL/PuLID seront réalisés dans les phases suivantes du plan.

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

Les dépendances de génération, volontairement non installées pendant le bootstrap, sont regroupées dans l'extra `inference` :

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

Tous les chemins sont configurés dans `config/default.yaml`. Les chemins relatifs de modèles sont résolus depuis `models_root`; les chemins relatifs d'artefacts sont résolus depuis la racine du dépôt.
