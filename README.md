# PuLID Python

Pipeline Python autonome pour générer des images SDXL conditionnées par une
identité PuLID. Le backend principal est Apple Silicon/MPS ; CUDA et CPU restent
pris en charge par la même architecture. Le projet ne dépend pas de ComfyUI.

Les phases 1 à 16 du plan sont implémentées : configuration et caches externes,
InsightFace, SDXL local, gestion mémoire, adaptateur PuLID v1.1, générateur,
CLI, benchmark, compatibilité CUDA, tests automatisés, erreurs métier et
documentation finale.

## 1. Prérequis

- Python 3.11 à 3.13 (Python 3.11 recommandé) ;
- `uv` pour créer l'environnement et installer le projet ;
- sur Mac, un Apple Silicon avec PyTorch/MPS ;
- en option, un GPU NVIDIA/CUDA pour le backend CUDA ;
- le SSD monté à l'emplacement configuré, par défaut
  `/Volumes/SSD/Documents/PuLID_models` ;
- les checkpoints et actifs décrits ci-dessous déjà présents sur ce SSD.

Le checkpoint SDXL par défaut est
`realvisxlV50_v50LightningBakedvae.safetensors`. Son VAE est intégré : aucun VAE
externe ne doit être configuré ou chargé.

## 2. Création de l'environnement

Sur macOS, la méthode recommandée crée `.venv` s'il est absent et met à jour un
environnement existant avec tous les extras, dont le serveur et le runtime GGUF :

```bash
./install_macos.sh
```

Le script ne télécharge aucun modèle. Il exige le GGUF local sous
`text_embedding/bge-m3-Q8_0.gguf`, force les caches Python et ML sous
`PULID_MODELS_ROOT`, puis vérifie PyTorch/MPS, `llama-cpp-python` et la
configuration du modèle. Il peut être relancé après chaque `git pull` ; les
dépendances déjà compatibles sont réutilisées.

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
```

Vérifier ensuite l'installation et la configuration sans charger les poids :

```bash
pulid-gen --version
pulid-gen doctor
python scripts/inspect_models.py --show-cache-env --fail-on-internal-cache
```

`doctor` vérifie notamment le montage du SSD, les checkpoints, AntelopeV2, les
configs SDXL, le runtime PuLID épinglé, EVA-CLIP, FaceXLib, les permissions, les
devices et les versions des dépendances critiques.

## 4. Arborescence des modèles

Les fichiers lourds restent sous `models_root`, jamais dans le dépôt :

```text
/Volumes/SSD/Documents/PuLID_models/
├── checkpoints/
│   ├── realvisxlV50_v50LightningBakedvae.safetensors
│   └── reaxl_v30.safetensors             # checkpoint SDXL alternatif
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

La préparation idempotente du code officiel PuLID se fait une seule fois :

```bash
python scripts/prepare_pulid.py
python scripts/prepare_pulid.py --check-only
```

IDFormer, EVA-CLIP et les processeurs d'attention sont ainsi réutilisés depuis
le SSD. Les poids EVA-CLIP et FaceXLib, lorsqu'ils doivent être acquis une
première fois, sont également conservés sous cette même racine externe.

## 5. Configuration du checkpoint SDXL

La configuration centrale est `config/default.yaml`. Les chemins de modèles
relatifs sont résolus depuis `models_root`, tandis que `outputs_dir` et
`identity_cache_dir` sont résolus depuis la racine du projet.

```yaml
models_root: /Volumes/SSD/Documents/PuLID_models

sdxl:
  checkpoint: checkpoints/realvisxlV50_v50LightningBakedvae.safetensors
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
Diffusers. La commande suivante prépare uniquement ces petits fichiers et
refuse les poids distants :

```bash
python scripts/prepare_sdxl_config.py
```

Il est possible de surcharger le fichier avec `--config` ou `PULID_CONFIG`, et
la racine externe avec `PULID_MODELS_ROOT`.

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

- sans option : BGE sur Metal/CUDA, sans offload SDXL ;
- `--partial` : BGE sur GPU, avec les deux encodeurs CLIP et le VAE SDXL sur CPU
  pendant les embeddings ; l'UNet et PuLID restent en VRAM ;
- `--full` : BGE sur GPU, avec le pipeline SDXL entièrement déchargé ;
- `--CPU` : BGE sur CPU, sans offload SDXL et avec concurrence CPU/GPU permise ;
- `--concurrent-cuda` : mode expérimental sans offload, autorisant BGE et SDXL
  à calculer simultanément sur CUDA.

```bash
./start_pulid_server.sh --partial
start_windows.bat --partial
start_windows.bat --concurrent-cuda
```

Les quatre options sont mutuellement exclusives. Les calculs GPU sont sérialisés
par défaut ; seul `--concurrent-cuda` retire ce verrou. Sur une carte de 12 Gio,
ce test peut provoquer un OOM pendant les pics mémoire : il suffit alors de
redémarrer sans cette option pour retrouver le comportement sûr. Avec
`--partial` ou `--full`, BGE reste chargé entre ses appels puis est fermé avant
la prochaine génération SDXL.

Sous Windows, `install_windows.bat` installe la wheel CUDA 13.0 épinglée de
`llama-cpp-python`, puis remplace uniquement son backend CPU auxiliaire par la
DLL portable issue de la wheel officielle de même version. Cela évite
`0xc000001d` sur les Core i9 sans AVX-512 tout en conservant le calcul BGE sur
CUDA. Le script vérifie ensuite un chargement et un embedding réels avec le
contexte complet de 8192 tokens ; aucun modèle n'est téléchargé.

Le contrat complet, les champs multipart, headers de réponse et exemples
TypeScript sont décrits dans [`API_FRONTEND_INTEGRATION.md`](API_FRONTEND_INTEGRATION.md).
La bascule de `rp-bot` depuis LM Studio est détaillée dans
[`RP_BOT_TEXT_EMBEDDING_INTEGRATION.md`](RP_BOT_TEXT_EMBEDDING_INTEGRATION.md).

## 11. Dépannage

Les erreurs CLI affichent leur type et une correction probable :

| Erreur | Cause probable | Correction |
|---|---|---|
| `ExternalDriveNotMountedError` | `models_root` ou son volume est absent | Monter le SSD puis relancer `pulid-gen doctor` |
| `ModelNotFoundError` | checkpoint/config/ONNX local absent | Corriger `config/default.yaml` ou le nom fourni à `--model` |
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
FaceXLib ou AntelopeV2 dans ce dépôt.
