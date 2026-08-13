# PuLID Python

Pipeline Python autonome pour SDXL et PuLID, conçu en priorité pour Apple Silicon/MPS. Les modèles et caches lourds restent sur le SSD externe ; seuls le code, les petites données d'identité et les résultats résident dans ce dépôt.

## État

Les phases 1 à 11 sont opérationnelles : bootstrap, configuration centralisée,
inspection, backends MPS/CUDA/CPU, cache d'identité AntelopeV2, SDXL local,
gestion mémoire, adaptateur PuLID v1.1, générateur haut niveau et CLI installable.
Le pipeline complet peut être utilisé depuis Python ou avec `pulid-gen`.

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
uv pip install -e '.[inference,pulid,dev]'
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

## Adaptateur PuLID v1.1 (phases 7 et 8)

L'adaptateur utilise l'architecture officielle : IDFormer produit 32 tokens
d'identité de dimension 2048 à partir d'ArcFace et d'EVA-CLIP, puis 70
processeurs PuLID sont injectés dans les cross-attentions de l'UNet SDXL. Le
générateur n'importe aucune classe interne de PuLID.

Le code officiel (IDFormer, EVA-CLIP et processeurs d'attention) est téléchargé
une seule fois à une révision Git épinglée dans :

```text
/Volumes/SSD/Documents/PuLID_models/sources/PuLID
```

La commande est idempotente et ne copie rien dans ce dépôt :

```bash
source .venv/bin/activate
python scripts/prepare_pulid.py
python scripts/prepare_pulid.py --check-only
```

EVA-CLIP et FaceXLib conservent leurs poids téléchargés automatiquement sous
`PuLID_models/huggingface/` et `PuLID_models/facexlib/weights/`. Ils ne sont donc
pas recréés ni retéléchargés à chaque exécution. Pour valider le checkpoint et
les traits de Noémie sans génération SDXL :

```bash
python scripts/test_pulid_adapter.py \
  --device mps \
  --reference inputs/noemie.webp \
  --offline
```

Pour vérifier l'injection dans le véritable UNet SDXL local :

```bash
python scripts/test_pulid_adapter.py --device mps --offline --apply-sdxl
```

Enfin, l'inspection peut afficher les caches effectifs et échouer si l'un d'eux
pointe hors de la racine du SSD :

```bash
python scripts/inspect_models.py --show-cache-env --fail-on-internal-cache
```

## Premier test PuLID complet (phase 9)

Le test complet utilise par défaut `inputs/noemie.webp`, charge tous les modèles
hors ligne, injecte PuLID dans RealVisXL puis écrit un PNG et son manifeste JSON
adjacent dans `outputs/` :

```bash
source .venv/bin/activate
python scripts/test_pulid.py \
  --reference inputs/noemie.webp \
  --prompt "cinematic portrait of a woman standing in Tokyo at night" \
  --seed 42 \
  --strength 0.8
```

Un autre checkpoint SDXL situé dans le même dossier que le modèle configuré
peut être sélectionné par son nom, sans l'extension `.safetensors` :

```bash
python scripts/test_pulid.py \
  --model reaxl_v30 \
  --reference inputs/noemie.webp \
  --prompt "cinematic portrait of a woman standing in Tokyo at night"
```

Sans `--model`, le checkpoint déclaré dans `config/default.yaml` reste utilisé.

Le scheduler et le CFG peuvent également être remplacés pour un test donné :

```bash
python scripts/test_pulid.py \
  --model reaxl_v30 \
  --method dpmpp_2m_sde_karras \
  --cfg 4.5
```

`--method` active ici DPM++ 2M SDE avec les sigmas Karras. Sans cette option,
le scheduler fourni par le checkpoint est conservé. Sans `--cfg`, le CFG reste
à sa valeur de base de `7.0`. L'ancien nom `--guidance-scale` reste accepté.

Le JSON contient la référence, les prompts, les paramètres d'inférence, les
checkpoints SDXL/PuLID, la révision du runtime officiel, le device, le dtype et
les durées effectives. Le VAE du checkpoint SDXL monofichier est utilisé.

## Générateur haut niveau (phase 10)

`ImageGenerator` charge les modèles au premier usage, réutilise le cache ArcFace,
prépare le conditionnement PuLID, sélectionne le device et sauvegarde
automatiquement un PNG et son JSON adjacent :

```python
from pulid_app.config import load_config
from pulid_app.pipeline import ImageGenerator

config = load_config()

with ImageGenerator(config, device="mps") as generator:
    identity = generator.encode_identity(
        "inputs/noemie.webp",
        identity_id="noemie",
    )
    result = generator.generate(
        prompt="cinematic portrait of a woman standing in Tokyo at night",
        identity=identity,
        seed=42,
        width=1024,
        height=1024,
        steps=20,
        identity_strength=0.8,
    )

print(result.png_path)
print(result.json_path)
```

Le contexte `with` garantit le cleanup des modèles. Sans `device`, le meilleur
backend disponible est choisi dans l'ordre CUDA, MPS, puis CPU. Les paramètres
`sampling_method="dpmpp_2m_sde_karras"` et `guidance_scale=...` restent
disponibles dans `generate()`.

## CLI installable (phase 11)

Après l'installation éditable du projet, la commande principale est disponible
dans l'environnement virtuel :

```bash
source .venv/bin/activate
pulid-gen --help
```

Le diagnostic complet ne charge aucun poids de génération :

```bash
pulid-gen doctor
```

Il vérifie le montage du SSD, les caches externes, les checkpoints, AntelopeV2,
les configurations SDXL, le runtime PuLID épinglé, FaceXLib, EVA-CLIP, les
permissions des sorties, MPS/CUDA et les versions des dépendances critiques.

Les commandes d'inspection et de cache sont également regroupées :

```bash
pulid-gen inspect-models --show-cache-env --fail-on-internal-cache
pulid-gen encode \
  --reference inputs/noemie.webp \
  --character noemie
```

Une génération complète s'exécute ainsi :

```bash
pulid-gen generate \
  --reference inputs/noemie.webp \
  --prompt "cinematic portrait of a woman standing in Tokyo at night" \
  --model reaxl_v30 \
  --method dpmpp_2m_sde_karras \
  --cfg 4.5 \
  --seed 42
```

La sous-commande `benchmark` est réservée mais retourne volontairement un code
non nul jusqu'à son implémentation dans la phase 12.

Tous les chemins sont configurés dans `config/default.yaml`. Les chemins relatifs de modèles sont résolus depuis `models_root`; les chemins relatifs d'artefacts sont résolus depuis la racine du dépôt.
