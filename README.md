# PuLID Python

Pipeline Python autonome pour SDXL et PuLID, conçu en priorité pour Apple Silicon/MPS. Les modèles et caches lourds restent sur le SSD externe ; seuls le code, les petites données d'identité et les résultats résident dans ce dépôt.

## État

Les phases 1 à 6 fournissent le bootstrap, la configuration centralisée,
l'inspection des modèles, la validation du backend PyTorch/MPS et l'extraction
faciale AntelopeV2 avec cache d'identité générique. La génération SDXL locale
est opérationnelle et son cycle de vie mémoire est explicite ; l'intégration de
PuLID sera réalisée dans les phases suivantes.

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

## Validation d'AntelopeV2 (phase 3)

La référence du projet est une image WebP :

```bash
source .venv/bin/activate
python scripts/test_insightface.py --image inputs/reference.webp
```

Pour enregistrer le résumé de la détection dans `cache/identity/` :

```bash
python scripts/test_insightface.py \
  --image inputs/reference.webp \
  --save-metadata
```

InsightFace utilise explicitement `CPUExecutionProvider` sur macOS. Si une image
contient plusieurs visages, il faut en sélectionner un avec `--face-index N`.

## Cache d'identité de Noémie (phase 4)

Le cache accepte les images JPEG, PNG et WebP, ainsi que BMP et TIFF. Il est
adressé par le SHA-256 du contenu de l'image : renommer un fichier ne provoque
donc pas un nouveau calcul.

```bash
source .venv/bin/activate
python scripts/cache_identity.py \
  --character noemie \
  --image inputs/reference.webp
```

Le premier appel crée une archive NPZ sous `cache/identity/`. Le deuxième appel
avec la même image relit cette archive sans charger les modèles ONNX ni recalculer
l'embedding. Utiliser `--force` pour imposer un recalcul.

## Génération SDXL locale (phase 5)

Le checkpoint monofichier contient l'UNet, les encodeurs de texte et le VAE, mais
pas les fichiers de tokenizer/configuration. Leur préparation est une opération
explicite qui télécharge environ 3 Mio de JSON/TXT vers le SSD et refuse tout
fichier de poids :

```bash
source .venv/bin/activate
python scripts/prepare_sdxl_config.py
```

La génération elle-même est ensuite strictement hors ligne (`local_files_only`) :

```bash
python scripts/test_sdxl.py \
  --prompt "portrait photo of a woman, tropical beach, studio lighting" \
  --seed 42
```

Les fichiers `outputs/sdxl_test_<timestamp>.png` et `.json` contiennent l'image
et ses paramètres effectifs. Le pipeline sélectionne MPS en FP16 et ne tente un
second chargement FP32 que pour une erreur FP16 compatible avec ce fallback ; un
manque de mémoire ne déclenche jamais ce second essai.

## Gestion mémoire (phase 6)

`MemoryManager` centralise les déplacements vers CPU/MPS/CUDA, le déchargement
des modules et le nettoyage des caches. Un cache accélérateur n'est vidé qu'une
fois après une ou plusieurs libérations ; `SDXLModel.close()` peut être appelé
plusieurs fois sans multiplier les appels à `empty_cache()`.

Le test suivant alloue une matrice de 64 Mio sur le meilleur backend disponible,
puis vérifie les compteurs avant et après libération, sans charger de modèle :

```bash
source .venv/bin/activate
python scripts/test_memory.py
```

Tous les chemins sont configurés dans `config/default.yaml`. Les chemins relatifs de modèles sont résolus depuis `models_root`; les chemins relatifs d'artefacts sont résolus depuis la racine du dépôt.
