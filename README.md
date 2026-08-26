# PuLID Python

Pipeline Python autonome pour générer des images SDXL conditionnées par une
identité PuLID. Le backend principal est Apple Silicon/MPS ; CUDA et CPU restent
pris en charge par la même architecture. Le projet ne dépend pas de ComfyUI.

Les phases 1 à 16 du plan sont implémentées : configuration et caches externes,
InsightFace, SDXL local, gestion mémoire, adaptateur PuLID v1.1, générateur,
CLI, benchmark, compatibilité CUDA, tests automatisés, erreurs métier et
documentation finale.

## 1. Prérequis

- une connexion Internet pour la première installation ;
- sur Mac, un Apple Silicon avec PyTorch/MPS ;
- sous Windows, un GPU NVIDIA, un pilote compatible CUDA 13 et, si InsightFace
  doit être compilé, Microsoft C++ Build Tools ;
- au moins 20 Go libres avec le SDXL Base proposé, davantage pour des checkpoints
  SDXL supplémentaires.

Les scripts installent `uv`, Python 3.11 et les dépendances si nécessaire. Les
modèles sont placés par défaut dans `PuLID_models/` à la racine du projet ; ce
dossier est ignoré par Git. Un SSD ou tout autre dossier accessible peut être
choisi à la place.

## 2. Création de l'environnement

La méthode recommandée crée `.venv` s'il est absent, installe le runtime adapté
à la plateforme, puis lance le CLI de préparation des modèles :

```bash
# macOS
./install_macos.sh

# Windows, depuis l'Explorateur ou cmd.exe
install_windows.bat
```

L'installateur demande d'abord s'il faut utiliser l'emplacement par défaut. En
cas de réponse négative, le chemin fourni peut désigner soit un dossier parent,
soit directement un dossier nommé `PuLID_models` ; dans ce dernier cas aucun
sous-dossier du même nom n'est ajouté.

Il vérifie ensuite les empreintes et installe ou répare PuLID v1.1,
AntelopeV2, EVA-CLIP, FaceXLib, BGE-M3, le snapshot de code PuLID et les
configurations/tokenizers SDXL. Les fichiers valides sont réutilisés. Pour SDXL,
il demande si un checkpoint existe déjà : l'utilisateur peut le déposer dans
`checkpoints/` et le sélectionner, ou accepter le téléchargement du modèle
officiel SDXL Base 1.0. À la fin, `doctor` doit réussir sans autre téléchargement
manuel.

L'installateur essaie d'abord la wheel macOS arm64 Metal précompilée. Si cette
archive n'est pas exploitable, il compile automatiquement `llama-cpp-python`
avec Accelerate et `GGML_METAL=ON` ; les outils de ligne de commande Xcode sont
alors requis. L'option serveur `--CPU` reste disponible avec ce même runtime.

Pour créer l'environnement manuellement depuis la racine du projet :

```bash
uv venv --python 3.11
source .venv/bin/activate
```

À chaque nouvelle session shell, réactiver l'environnement avec
`source .venv/bin/activate`.

## 3. Installation

L'installation manuelle complète pour l'inférence, le serveur, les embeddings
et le développement est :

```bash
uv pip install -e '.[inference,pulid,server,embeddings,dev]'
pulid-install
```

Pour automatiser ce CLI avec un emplacement et le modèle SDXL officiel :

```bash
pulid-install --models-root /chemin/parent --sdxl download
```

`--models-root` accepte également le chemin final `.../PuLID_models`.
`--sdxl existing` exige qu'au moins un `.safetensors` soit déjà présent dans
`checkpoints/`, et `--force-configs` force l'actualisation des petites
configurations SDXL.

Vérifier ensuite l'installation et la configuration sans charger les poids :

```bash
pulid-gen --version
pulid-gen doctor
python scripts/inspect_models.py --show-cache-env --fail-on-internal-cache
```

`doctor` vérifie notamment `models_root`, les checkpoints, AntelopeV2, les
configs SDXL, le runtime PuLID épinglé, EVA-CLIP, FaceXLib, les permissions, les
devices et les versions des dépendances critiques.

## 4. Arborescence des modèles

Les fichiers lourds restent sous `models_root`, dont le dossier est ignoré par
Git :

```text
<models_root>/
├── checkpoints/
│   ├── sd_xl_base_1.0.safetensors         # si le téléchargement est accepté
│   └── <checkpoint-utilisateur>.safetensors
├── pulid_v1.1.safetensors
├── antelopev2/
│   ├── 1k3d68.onnx
│   ├── 2d106det.onnx
│   ├── genderage.onnx
│   ├── glintr100.onnx
│   └── scrfd_10g_bnkps.onnx
├── sdxl/stable-diffusion-xl-base-1.0-config/
│   ├── model_index.json
│   ├── scheduler/
│   ├── text_encoder/
│   ├── text_encoder_2/
│   ├── tokenizer/
│   ├── tokenizer_2/
│   ├── unet/
│   └── vae/
├── sources/PuLID/                        # code officiel à révision épinglée
│   ├── eva_clip/
│   └── pulid/
├── facexlib/weights/
├── text_embedding/
│   └── bge-m3-Q8_0.gguf
├── huggingface/
├── torch/
└── other/
```

La réparation complète est idempotente et peut être relancée après une mise à
jour du dépôt :

```bash
pulid-install
```

Les scripts historiques `prepare_pulid.py` et `prepare_sdxl_config.py` restent
disponibles pour ne préparer qu'un composant.

## 5. Configuration du checkpoint SDXL

`config/default.yaml` contient les valeurs du dépôt. L'installateur génère
`config/local.yaml`, ignoré par Git, avec le `models_root`, le checkpoint SDXL
sélectionné et le device de la plateforme. Cette configuration locale est
chargée automatiquement ; une option `--config` ou `PULID_CONFIG` explicite
reste prioritaire.

```yaml
models_root: /chemin/choisi/PuLID_models

sdxl:
  checkpoint: checkpoints/sd_xl_base_1.0.safetensors
  config_dir: sdxl/stable-diffusion-xl-base-1.0-config

device:
  preferred: mps
  dtype: float16
  offload_strategy: none

text_embedding:
  checkpoint: text_embedding/bge-m3-Q8_0.gguf
  model_id: text-embedding-bge-m3
  dimensions: 1024
  context_size: 8192
  batch_size: 8192
  threads: 0  # automatique
```

Un monofichier SDXL ne contient pas les JSON et tokenizers attendus par
Diffusers. `pulid-install` actualise uniquement ces petits fichiers et refuse
les poids distants dans leur dossier de configuration. La commande spécialisée
reste disponible :

```bash
python scripts/prepare_sdxl_config.py
```

La racine peut aussi être imposée à un script d'installation avec
`PULID_MODELS_ROOT` ; cette variable évite alors la question interactive.

## 6. Test MPS

```bash
python scripts/test_mps.py
python scripts/test_memory.py
```

Le premier script affiche PyTorch, MPS/CUDA, le dtype et la mémoire détectée,
puis exécute un petit calcul MPS. Le second vérifie le déplacement et le
nettoyage mémoire sans charger de checkpoint lourd. La sélection automatique
suit l'ordre CUDA, MPS, CPU.

## 7. Test InsightFace

La référence de test est `inputs/noemie.webp`. JPEG, PNG, WebP, BMP et TIFF sont
acceptés.

```bash
python scripts/test_insightface.py \
  --image inputs/noemie.webp \
  --save-metadata

python scripts/cache_identity.py \
  --character noemie \
  --image inputs/noemie.webp
```

Sur macOS, InsightFace utilise `CPUExecutionProvider`. Le cache ArcFace est
adressé par le SHA-256 du contenu ; renommer l'image ne déclenche donc pas un
nouveau calcul. Pour une photo comportant plusieurs visages, fournir
`--face-index N`. Utiliser `--force` pour recalculer le cache.

## 8. Test SDXL

Le chargement vise toujours un fichier `.safetensors` local explicite et utilise
`local_files_only=True` : aucun téléchargement implicite de SDXL n'est permis.

```bash
python scripts/test_sdxl.py \
  --prompt "portrait photo of a woman, tropical beach, studio lighting" \
  --seed 42
```

Le test écrit `outputs/sdxl_test_<timestamp>.png` et son JSON adjacent. Sur MPS,
le pipeline utilise FP16 et n'essaie FP32 que pour une incompatibilité de dtype
admissible ; un manque de mémoire ne déclenche pas ce second essai.

Les prompts positifs et négatifs utilisent l'encodage Diffusers natif jusqu'à
75 jetons CLIP utiles. Entre 76 et 255 jetons, ils sont segmentés en blocs CLIP
dont les embeddings sont concaténés sans troncature. Au-delà de 255 jetons avec
l'un des deux tokenizers SDXL, la génération échoue explicitement avec
`PromptTooLongError`.

## 9. Test PuLID

Valider d'abord l'adaptateur, puis son injection dans le véritable UNet :

```bash
python scripts/test_pulid_adapter.py \
  --device mps \
  --reference inputs/noemie.webp \
  --offline

python scripts/test_pulid_adapter.py \
  --device mps \
  --reference inputs/noemie.webp \
  --offline \
  --apply-sdxl
```

Le test complet historique reste disponible :

```bash
python scripts/test_pulid.py \
  --reference inputs/noemie.webp \
  --prompt "cinematic portrait of a woman standing in Tokyo at night" \
  --seed 42 \
  --strength 0.8
```

Pour employer ReaXL, fournir seulement le nom sans extension :

```bash
python scripts/test_pulid.py \
  --model reaxl_v30 \
  --method dpmpp_2m_sde \
  --sigmas karras \
  --cfg 4.5
```

La méthode de sampling et la courbe de sigmas sont indépendantes. Sans
`--model`, `--method` ou `--cfg`, le modèle, le scheduler et le CFG
préconfigurés restent inchangés ; `--sigmas` vaut `normal` par défaut.

## 10. Génération finale

La CLI installable encode la référence, applique PuLID, génère l'image et écrit
automatiquement ses métadonnées :

```bash
pulid-gen generate \
  --reference inputs/noemie.webp \
  --character noemie \
  --prompt "cinematic portrait of a woman standing in Tokyo at night" \
  --model reaxl_v30 \
  --method dpmpp_2m_sde \
  --sigmas karras \
  --cfg 4.5 \
  --strength 0.8 \
  --steps 20 \
  --seed 42
```

Le checkpoint alternatif doit se trouver dans `PuLID_models/checkpoints/`, à
côté du modèle configuré ; passer son nom sans `.safetensors`. Omettre
`--model` utilise RealVisXL par défaut.

L'API Python de haut niveau fournit le même cycle de vie :

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
        steps=20,
        identity_strength=0.8,
        sampling_method="dpmpp_2m_sde",
        sigma_schedule="karras",
    )

print(result.png_path)
print(result.json_path)
```

Sur une machine CUDA disposant d'une VRAM limitée, l'offload SDXL est opt-in :

```bash
pulid-gen generate \
  --device cuda \
  --offload model_cpu_offload \
  --reference inputs/noemie.webp \
  --prompt "portrait photo"
```

`model_cpu_offload` est refusé sur MPS et CPU. La valeur par défaut `none`
préserve le comportement MPS actuel.

## Benchmark et tests automatisés

Le benchmark exécute des runs froids avec la même seed et sépare neuf durées :
chargement SDXL, PuLID, InsightFace, extraction d'identité, préparation du
prompt, diffusion, VAE, sauvegarde et total.

```bash
pulid-gen benchmark \
  --reference inputs/noemie.webp \
  --prompt "cinematic portrait" \
  --runs 3 \
  --model reaxl_v30 \
  --method dpmpp_2m_sde \
  --sigmas karras \
  --cfg 4.5
```

Les catégories pytest sont `unit`, `integration`, `slow` et `gpu` :

```bash
pytest -m unit
pytest -m integration
pytest
```

Les tests d'intégration sont opt-in afin que les tests normaux ne chargent ni le
SSD ni les modèles lourds :

```bash
PULID_RUN_INTEGRATION=1 pytest -m integration -k insightface
PULID_RUN_SLOW=1 pytest -m 'integration and slow'
```

## Serveur HTTP pour frontend

Le serveur local expose les routes SDXL `GET /models` et `POST /generate`, ainsi
que les routes OpenAI compatibles `GET /v1/models` et `POST /v1/embeddings`.
La génération HTTP renvoie directement un PNG, n'écrit ni output ni JSON, et
crée ou réutilise le petit cache ArcFace sous `cache/identity/` :

```bash
uv pip install -e '.[inference,pulid,server,embeddings]'
pulid-server --device mps --cors-origin http://localhost:3000
```

Avec `--device cuda`, le pipeline SDXL reste chargé en VRAM entre deux requêtes
utilisant le même checkpoint. Il est déchargé lorsqu'une requête sélectionne un
autre modèle ou lorsque le serveur s'arrête. MPS et CPU conservent le nettoyage
après chaque génération.

Le GGUF BGE-M3 est chargé paresseusement. Le serveur propose cinq modes, sans
jamais modifier sa fenêtre de 8192 tokens :

- sans option : BGE et SDXL peuvent calculer simultanément sur CUDA, sans
  offload ; sur Metal, les calculs restent sérialisés ;
- `--serialized-cuda` : BGE et SDXL sur GPU avec un verrou commun, sans offload ;
- `--partial` : BGE sur GPU, avec les deux encodeurs CLIP et le VAE SDXL sur CPU
  pendant les embeddings ; l'UNet et PuLID restent en VRAM ;
- `--full` : BGE sur GPU, avec le pipeline SDXL entièrement déchargé ;
- `--CPU` : BGE sur CPU, sans offload SDXL et avec concurrence CPU/GPU permise.

```bash
./start_pulid_server.sh --partial
start_windows.bat --partial
start_windows.bat --serialized-cuda
```

Les quatre options sont mutuellement exclusives. Sur CUDA, les calculs sont
concurrents par défaut ; `--serialized-cuda` rétablit le verrou commun si une
configuration provoque un OOM pendant les pics mémoire. Avec `--partial` ou
`--full`, BGE reste chargé entre ses appels puis est fermé avant la prochaine
génération SDXL.

Sous Windows, `install_windows.bat` installe la wheel CUDA 13.0 épinglée de
`llama-cpp-python`, puis remplace uniquement son backend CPU auxiliaire par la
DLL portable issue de la wheel officielle de même version. Cela évite
`0xc000001d` sur les Core i9 sans AVX-512 tout en conservant le calcul BGE sur
CUDA. Le script vérifie ensuite un chargement et un embedding réels avec le
contexte complet de 8192 tokens. Les modèles manquants sont préparés juste avant
ces contrôles par le même CLI que sous macOS.

Le contrat complet, les champs multipart, headers de réponse et exemples
TypeScript sont décrits dans [`API_FRONTEND_INTEGRATION.md`](API_FRONTEND_INTEGRATION.md).
La bascule de `rp-bot` depuis LM Studio est détaillée dans
[`RP_BOT_TEXT_EMBEDDING_INTEGRATION.md`](RP_BOT_TEXT_EMBEDDING_INTEGRATION.md).

## 11. Dépannage

Les erreurs CLI affichent leur type et une correction probable :

| Erreur | Cause probable | Correction |
|---|---|---|
| `ExternalDriveNotMountedError` | `models_root` ou son volume est absent | Rendre le dossier accessible puis relancer `pulid-install` et `pulid-gen doctor` |
| `ModelNotFoundError` | checkpoint/config/ONNX local absent | Relancer `pulid-install`, ou corriger `config/local.yaml` / le nom fourni à `--model` |
| `FaceNotDetectedError` | aucun visage exploitable | Utiliser une référence nette, de face et suffisamment grande |
| `MultipleFacesDetectedError` | plusieurs visages détectés | Ajouter `--face-index N` |
| `PromptTooLongError` | prompt positif ou négatif supérieur à 255 jetons CLIP utiles | Raccourcir le texte en conservant les concepts prioritaires |
| `UnsupportedDeviceError` | backend ou offload incompatible | Choisir `mps`, `cuda` ou `cpu`; réserver l'offload à CUDA |
| `ModelLoadError` | dépendance, provider, poids ou mémoire | Lancer `doctor`, vérifier les versions et réduire la charge mémoire |
| `GenerationError` | paramètre ou étape d'inférence en échec | Lire la cause affichée, vérifier dimensions/steps/CFG puis réessayer |
| `EmbeddingError` | réponse ou calcul GGUF invalide | Vérifier le GGUF, `llama-cpp-python`, la longueur du texte et relancer |
| `Failed to load ... llama.dll` | DLL CUDA de PyTorch ou `nvcudart_hybrid64.dll` du pilote NVIDIA absente du chemin Windows | Mettre à jour le pilote NVIDIA puis relancer `install_windows.bat`; le serveur ajoute automatiquement `torch\lib` et le dossier NVIDIA du `DriverStore` à `PATH` et à la recherche sécurisée de DLL |
| `Windows Error 0xc000001d` | backend CPU auxiliaire de la wheel CUDA compilé avec AVX-512 sur un CPU incompatible | Faire un `git pull`, fermer le serveur et relancer `install_windows.bat`; le script installe automatiquement la DLL CPU portable sans désactiver CUDA |

Contrôles utiles :

```bash
pulid-gen doctor
pulid-gen inspect-models --show-cache-env --fail-on-internal-cache
python scripts/verify_text_embedding.py --device cuda
```

Les dimensions doivent être positives et divisibles par 8, la seed positive ou
nulle, les steps strictement positifs, et le CFG positif ou nul.

## 12. Emplacement des outputs

Les sorties normales sont écrites sous `outputs/` avec un même stem :

```text
outputs/
├── pulid_<timestamp>.png
├── pulid_<timestamp>.json
└── benchmarks/
    ├── benchmark_<timestamp>.json
    └── runs/
        ├── benchmark_run_001_<timestamp>.png
        └── benchmark_run_001_<timestamp>.json
```

Le manifeste JSON conserve les références, prompts, seed, dimensions, steps,
CFG, méthode de sampling, force PuLID, checkpoints, device, dtype, VAE intégré
et durées effectives.

## 13. Emplacement des caches

Seul le petit cache d'identité est conservé dans le projet :

```text
cache/identity/*.npz
```

Tous les caches lourds sont forcés sous `models_root` avant les imports ML :

| Variable | Emplacement sous `PuLID_models` |
|---|---|
| `HF_HOME` | `huggingface/` |
| `HUGGINGFACE_HUB_CACHE` | `huggingface/hub/` |
| `TRANSFORMERS_CACHE` | `huggingface/transformers/` |
| `TORCH_HOME` | `torch/` |
| `XDG_CACHE_HOME` | `other/` |
| `MPLCONFIGDIR` | `other/matplotlib/` |

Il ne faut ni déplacer ni recopier les checkpoints, EVA-CLIP, IDFormer,
FaceXLib ou AntelopeV2 dans les modules applicatifs ; l'installateur les garde
tous sous le `models_root` sélectionné.
