# Plan d’implémentation — PuLID + SDXL en Python pur sur macOS

## 0. Objectif du projet

Construire un pipeline Python autonome, sans ComfyUI, permettant de :

1. charger un checkpoint SDXL local ;
2. charger PuLID v1.1 et les modèles InsightFace/AntelopeV2 depuis un SSD externe ;
3. encoder l’identité d’un personnage depuis une ou plusieurs images ;
4. générer une image SDXL conditionnée par cette identité ;
5. utiliser Apple Metal / MPS sur un Mac M1 Max ;
6. conserver **tous les modèles, poids et caches volumineux** dans :

```text
/Volumes/SSD/Documents/PuLID_models
```

7. conserver **les images générées, logs et artefacts d’exécution dans le dossier du projet**.

Le projet doit rester simple à exécuter, modulaire, testable, et suffisamment abstrait pour pouvoir remplacer PuLID plus tard par PhotoMaker, InstantID ou une autre méthode d’identité.

---

# 1. Contraintes importantes

## Matériel cible initial

```text
Apple Silicon
Mac M1 Max
32 Go de mémoire unifiée
Backend PyTorch : MPS
```

Le projet devra toutefois éviter les hypothèses trop spécifiques à macOS afin de pouvoir être utilisé plus tard sur :

```text
NVIDIA RTX 4070 12 Go
CUDA
64 Go de RAM système
```

## Stockage des modèles

Aucun modèle lourd ne doit être copié dans le repo.

Utiliser comme racine :

```text
/Volumes/SSD/Documents/PuLID_models
```

Cette racine doit contenir les modèles téléchargés manuellement ainsi que les futurs caches Hugging Face / Torch / autres bibliothèques.

Exemple de structure cible :

```text
/Volumes/SSD/Documents/PuLID_models/
├── sdxl/
│   └── <checkpoint-sdxl>.safetensors
│
├── pulid/
│   └── pulid_v1.1.safetensors
│
├── insightface/
│   └── antelopev2/
│       ├── 1k3d68.onnx
│       ├── 2d106det.onnx
│       ├── genderage.onnx
│       ├── glintr100.onnx
│       └── scrfd_10g_bnkps.onnx
│
├── huggingface/
│   ├── hub/
│   └── transformers/
│
├── torch/
│
└── other/
```

Les noms exacts présents sur disque peuvent différer. Le code doit donc centraliser les chemins dans une configuration et ne jamais les hardcoder dans les modules métier.

---

# 2. Structure recommandée du repo

Créer un repo du type :

```text
pulid-python/
├── README.md
├── pyproject.toml
├── .gitignore
├── .env.example
│
├── config/
│   └── default.yaml
│
├── src/
│   └── pulid_app/
│       ├── __init__.py
│       ├── config.py
│       ├── device.py
│       ├── paths.py
│       │
│       ├── models/
│       │   ├── __init__.py
│       │   ├── sdxl.py
│       │   ├── identity_encoder.py
│       │   └── pulid_adapter.py
│       │
│       ├── pipeline/
│       │   ├── __init__.py
│       │   ├── generator.py
│       │   └── memory.py
│       │
│       ├── io/
│       │   ├── __init__.py
│       │   ├── images.py
│       │   └── metadata.py
│       │
│       └── cli.py
│
├── scripts/
│   ├── inspect_models.py
│   ├── test_mps.py
│   ├── test_insightface.py
│   ├── test_sdxl.py
│   ├── test_pulid.py
│   └── generate.py
│
├── tests/
│   ├── test_paths.py
│   ├── test_config.py
│   └── test_device.py
│
├── inputs/
│   └── .gitkeep
│
├── outputs/
│   └── .gitkeep
│
└── cache/
    └── identity/
        └── .gitkeep
```

Les images générées devront être écrites ici :

```text
<repo>/outputs/
```

Les embeddings d’identité calculés peuvent être stockés ici :

```text
<repo>/cache/identity/
```

Ils sont suffisamment petits pour rester avec le projet.

---

# 3. Configuration des chemins

Créer un fichier :

```text
config/default.yaml
```

Exemple :

```yaml
models_root: /Volumes/SSD/Documents/PuLID_models

sdxl:
  checkpoint: /Volumes/SSD/Documents/PuLID_models/sdxl/MODEL.safetensors

pulid:
  checkpoint: /Volumes/SSD/Documents/PuLID_models/pulid/pulid_v1.1.safetensors

insightface:
  model_root: /Volumes/SSD/Documents/PuLID_models/insightface
  model_name: antelopev2

outputs_dir: ./outputs
identity_cache_dir: ./cache/identity

device:
  preferred: mps
  dtype: float16
```

Ne jamais supposer que `MODEL.safetensors` est réellement le nom du checkpoint.

Le premier script d’inspection doit afficher les checkpoints détectés et permettre à l’utilisateur de reporter le bon nom dans la configuration.

---

# 4. Variables d’environnement obligatoires

Au démarrage du CLI, configurer les variables de cache pour empêcher Hugging Face ou Torch d’écrire de gros fichiers dans le disque interne.

Exemple :

```bash
export HF_HOME="/Volumes/SSD/Documents/PuLID_models/huggingface"
export HUGGINGFACE_HUB_CACHE="/Volumes/SSD/Documents/PuLID_models/huggingface/hub"
export TRANSFORMERS_CACHE="/Volumes/SSD/Documents/PuLID_models/huggingface/transformers"
export TORCH_HOME="/Volumes/SSD/Documents/PuLID_models/torch"
```

Ajouter éventuellement :

```bash
export XDG_CACHE_HOME="/Volumes/SSD/Documents/PuLID_models/other"
```

La configuration de ces variables doit se faire **avant les imports qui peuvent déclencher des téléchargements**.

Créer une fonction dédiée :

```python
configure_external_model_caches()
```

Cette fonction doit être appelée au tout début du programme.

---

# 5. Phase 1 — Bootstrap du projet

## Tâches

- créer la structure du repo ;
- créer un environnement Python ;
- ajouter `pyproject.toml` ;
- ajouter les dépendances minimales ;
- créer `.gitignore` ;
- créer le système de configuration ;
- créer la gestion centralisée des chemins.

## Dépendances probables

Commencer avec :

```text
torch
torchvision
diffusers
transformers
accelerate
safetensors
huggingface_hub
Pillow
numpy
opencv-python
onnxruntime
insightface
pyyaml
rich
```

Ne pas ajouter de dépendance inutile tant qu’elle n’est pas requise.

PuLID peut demander des dépendances additionnelles. Les ajouter uniquement lorsqu’elles sont réellement nécessaires.

## Critère d’acceptation

La commande :

```bash
python scripts/inspect_models.py
```

doit :

1. vérifier que `/Volumes/SSD/Documents/PuLID_models` existe ;
2. afficher l’espace disque libre ;
3. localiser le checkpoint PuLID ;
4. localiser AntelopeV2 ;
5. afficher tous les `.safetensors` candidats pour SDXL ;
6. vérifier que `outputs/` est accessible en écriture.

---

# 6. Phase 2 — Détection du device

Créer :

```text
src/pulid_app/device.py
```

API minimale :

```python
def get_best_device() -> str:
    ...

def get_default_dtype(device: str):
    ...
```

Ordre de priorité :

```text
CUDA
MPS
CPU
```

Pour le Mac :

```python
torch.backends.mps.is_available()
```

doit être vérifié.

Prévoir également une fonction :

```python
print_device_report()
```

qui affiche :

- device sélectionné ;
- version PyTorch ;
- disponibilité MPS ;
- disponibilité CUDA ;
- dtype sélectionné ;
- mémoire unifiée / informations disponibles si récupérables proprement.

## Critère d’acceptation

```bash
python scripts/test_mps.py
```

doit exécuter une petite opération tensorielle sur MPS sans erreur.

---

# 7. Phase 3 — Validation d’InsightFace / AntelopeV2

Avant PuLID ou SDXL, valider l’extraction faciale seule.

Créer :

```text
src/pulid_app/models/identity_encoder.py
```

Première version :

```python
class IdentityEncoder:
    def __init__(self, ...):
        ...

    def load(self):
        ...

    def detect(self, image):
        ...

    def encode(self, image):
        ...
```

Sur macOS, les modèles ONNX InsightFace doivent probablement utiliser :

```text
CPUExecutionProvider
```

Ne pas supposer que les modèles ONNX peuvent utiliser MPS.

Le coût est acceptable : l’extraction d’identité est ponctuelle et peut être mise en cache.

## Test

```bash
python scripts/test_insightface.py \
  --image inputs/reference.jpg
```

Le script doit :

1. charger AntelopeV2 depuis le SSD ;
2. détecter le visage ;
3. afficher le nombre de visages ;
4. afficher bounding box et score ;
5. extraire l’embedding ;
6. afficher sa shape et sa norme ;
7. sauvegarder éventuellement les métadonnées dans :

```text
cache/identity/
```

## Gestion des erreurs

Refuser proprement :

- aucune face ;
- plusieurs faces sans option explicite ;
- image non lisible ;
- chemin de modèle incorrect.

---

# 8. Phase 4 — Cache d’identité

Créer un format sérialisable pour l’identité.

Exemple :

```python
@dataclass
class CharacterIdentity:
    id: str
    source_images: list[str]
    face_embedding: np.ndarray
    metadata: dict
```

Ne pas lier cette classe directement à PuLID.

Elle doit représenter une identité générique.

API souhaitée :

```python
identity = identity_encoder.encode_image("inputs/alice.jpg")

identity.save("cache/identity/alice.npz")

identity = CharacterIdentity.load("cache/identity/alice.npz")
```

Créer un hash du contenu de l’image afin d’éviter de recalculer inutilement l’embedding.

## Critère d’acceptation

Deux appels avec la même image doivent réutiliser le cache.

---

# 9. Phase 5 — SDXL seul

Avant PuLID, valider que le checkpoint SDXL local fonctionne correctement sous MPS.

Créer :

```text
src/pulid_app/models/sdxl.py
```

Supporter en priorité un checkpoint `.safetensors` unique avec :

```python
StableDiffusionXLPipeline.from_single_file(...)
```

Ne pas télécharger de modèle SDXL depuis Internet.

Charger le checkpoint depuis :

```text
/Volumes/SSD/Documents/PuLID_models/...
```

Première configuration :

```text
device = mps
dtype = float16
resolution = 1024x1024
steps = 20
batch = 1
```

Si FP16 provoque des problèmes MPS, prévoir un fallback contrôlé.

Utiliser :

```python
torch.inference_mode()
```

pendant la génération.

## Test

```bash
python scripts/test_sdxl.py \
  --prompt "portrait photo of a woman, tropical beach, studio lighting" \
  --seed 42
```

Résultat attendu :

```text
outputs/sdxl_test_<timestamp>.png
```

Créer également un fichier JSON adjacent :

```text
outputs/sdxl_test_<timestamp>.json
```

avec :

```json
{
  "prompt": "...",
  "seed": 42,
  "steps": 20,
  "width": 1024,
  "height": 1024,
  "checkpoint": "...",
  "device": "mps"
}
```

---

# 10. Phase 6 — Gestion mémoire de base

Créer :

```text
src/pulid_app/pipeline/memory.py
```

API suggérée :

```python
class MemoryManager:
    def unload(self, module):
        ...

    def move_to_device(self, module, device):
        ...

    def cleanup(self):
        ...
```

Sur MPS, utiliser si pertinent :

```python
torch.mps.empty_cache()
```

Sur CUDA :

```python
torch.cuda.empty_cache()
```

Ne pas multiplier les appels `empty_cache()` inutilement.

Le pipeline doit privilégier des phases déterministes :

```text
Identity encoder
↓
libération
↓
Text encoding / SDXL
↓
PuLID
↓
VAE decode
```

Pour le premier POC Mac, privilégier la stabilité à l’optimisation extrême.

---

# 11. Phase 7 — Intégration PuLID v1.1

Ne pas commencer par réécrire PuLID.

Étudier d’abord l’implémentation officielle afin d’identifier précisément :

- architecture des modules ;
- poids attendus ;
- extracteur EVA-CLIP utilisé ;
- preprocessing ;
- points d’injection dans SDXL ;
- format des embeddings ;
- éventuels modules FaceXLib ;
- modifications du pipeline / attention processors.

Créer ensuite :

```text
src/pulid_app/models/pulid_adapter.py
```

API cible :

```python
class PuLIDAdapter:
    def __init__(self, checkpoint_path, ...):
        ...

    def load(self):
        ...

    def prepare_identity(self, image, face_embedding=None):
        ...

    def apply(self, pipeline):
        ...

    def set_identity(self, identity_features, strength=1.0):
        ...

    def clear_identity(self):
        ...
```

## Important

Le code métier du générateur ne doit pas dépendre directement des classes internes de PuLID.

---

# 12. Phase 8 — Téléchargements automatiques supplémentaires

Certains composants de PuLID peuvent exiger EVA-CLIP, FaceXLib ou d’autres poids.

Tous les téléchargements supplémentaires doivent être redirigés vers :

```text
/Volumes/SSD/Documents/PuLID_models
```

Il est interdit de laisser les bibliothèques remplir silencieusement :

```text
~/.cache/huggingface
~/.cache/torch
```

Créer une commande :

```bash
python scripts/inspect_models.py --show-cache-env
```

qui affiche les emplacements effectifs des caches.

Créer aussi une option :

```bash
python scripts/inspect_models.py --fail-on-internal-cache
```

qui échoue si les variables indiquent le disque interne.

---

# 13. Phase 9 — Premier test PuLID complet

Créer :

```bash
python scripts/test_pulid.py \
  --reference inputs/reference.webp \
  --prompt "cinematic portrait of a woman standing in Tokyo at night" \
  --model reaxl_v30 \
  --method dpmpp_2m_sde_karras \
  --cfg 4 \
  --seed 42 \
  --strength 0.8
```

Pipeline :

```text
reference.jpg
      │
      ▼
InsightFace
      │
      ▼
face embedding
      │
      ├── EVA-CLIP / features nécessaires à PuLID
      │
      ▼
PuLID conditioning
      │
prompt ─────► SDXL
      │
      ▼
denoising
      │
      ▼
VAE
      │
      ▼
PNG
```

Résultat :

```text
outputs/pulid_<timestamp>.png
outputs/pulid_<timestamp>.json
```

Le JSON doit inclure au minimum :

```json
{
  "reference_image": "...",
  "prompt": "...",
  "negative_prompt": "...",
  "seed": 42,
  "steps": 20,
  "guidance_scale": 7.0,
  "identity_strength": 0.8,
  "width": 1024,
  "height": 1024,
  "sdxl_checkpoint": "...",
  "pulid_checkpoint": "...",
  "device": "mps"
}
```

---

# 14. Phase 10 — Générateur haut niveau

Créer :

```text
src/pulid_app/pipeline/generator.py
```

API souhaitée :

```python
generator = ImageGenerator(config)

identity = generator.encode_identity(
    "inputs/reference.jpg"
)

result = generator.generate(
    prompt="...",
    identity=identity,
    seed=42,
    width=1024,
    height=1024,
    steps=20,
    identity_strength=0.8,
)
```

Le générateur doit gérer :

- chargement lazy des modèles ;
- sélection device ;
- seed reproductible ;
- création automatique des noms de fichiers ;
- sauvegarde PNG ;
- sauvegarde JSON ;
- cache d’identité ;
- cleanup mémoire ;
- erreurs lisibles.

---

# 15. Phase 11 — CLI propre

Créer une commande installable :

```bash
pulid-gen generate \
  --reference inputs/alice.jpg \
  --prompt "..." \
  --seed 42
```

Sous-commandes souhaitées :

```text
pulid-gen doctor
pulid-gen inspect-models
pulid-gen encode
pulid-gen generate
pulid-gen benchmark
```

## `doctor`

Doit vérifier :

- SSD monté ;
- modèles présents ;
- cache configuré ;
- MPS/CUDA disponible ;
- accès lecture modèles ;
- accès écriture outputs ;
- versions des dépendances critiques.

---

# 16. Phase 12 — Benchmark

Créer un benchmark reproductible.

Commande :

```bash
pulid-gen benchmark \
  --reference inputs/reference.jpg \
  --prompt "portrait photo" \
  --runs 3
```

Mesurer séparément :

```text
temps chargement SDXL
temps chargement PuLID
temps chargement InsightFace
temps extraction identité
temps préparation prompt
temps diffusion
temps VAE
temps sauvegarde
temps total
```

Sur CUDA, ajouter plus tard :

```text
peak VRAM
allocated VRAM
reserved VRAM
```

Sur MPS, enregistrer les métriques accessibles proprement.

Produire :

```text
outputs/benchmarks/benchmark_<timestamp>.json
```

---

# 17. Phase 13 — Compatibilité CUDA future

Ne pas optimiser CUDA immédiatement, mais garder l’architecture compatible.

Prévoir que le futur backend CUDA pourra utiliser :

```python
pipe.enable_model_cpu_offload()
```

ou une stratégie manuelle.

Ne jamais appeler une API CUDA sans vérifier :

```python
device == "cuda"
```

Les composants doivent fonctionner selon ce principe :

```text
Mac :
MPS + RAM unifiée

PC :
CUDA 12 Go + RAM 64 Go
```

Le futur objectif sur RTX 4070 sera de conserver principalement le modèle de diffusion actif en VRAM et d’offloader les autres composants.

---

# 18. Phase 14 — Tests automatisés

Les tests unitaires ne doivent pas charger SDXL sauf tests explicitement marqués `slow`.

Catégories :

```text
unit
integration
slow
gpu
```

Tests rapides :

- parsing config ;
- résolution chemins ;
- détection device ;
- nommage fichiers ;
- cache identité ;
- métadonnées ;
- validation paramètres.

Tests intégration :

- InsightFace sur image exemple ;
- SDXL génération minimale ;
- PuLID génération minimale.

Utiliser `pytest`.

---

# 19. Phase 15 — Gestion des erreurs

Créer des exceptions explicites :

```python
ModelNotFoundError
ExternalDriveNotMountedError
FaceNotDetectedError
MultipleFacesDetectedError
UnsupportedDeviceError
ModelLoadError
GenerationError
```

Les erreurs CLI doivent être lisibles et indiquer la correction probable.

Exemple :

```text
ExternalDriveNotMountedError:
Expected model directory:
  /Volumes/SSD/Documents/PuLID_models

Check that the SSD is mounted before running generation.
```

---

# 20. Phase 16 — README

Le README final doit expliquer :

1. prérequis ;
2. création environnement ;
3. installation ;
4. arborescence des modèles ;
5. configuration du checkpoint SDXL ;
6. test MPS ;
7. test InsightFace ;
8. test SDXL ;
9. test PuLID ;
10. génération finale ;
11. dépannage ;
12. emplacement des outputs ;
13. emplacement des caches.

Ne jamais demander à l’utilisateur de déplacer les modèles vers le repo.

---

# 21. Règles de travail pour les agents Codex

Les agents doivent suivre ces règles.

## Règle 1 — Une phase à la fois

Ne pas implémenter plusieurs grosses phases simultanément.

À la fin de chaque phase :

```text
1. exécuter les tests ;
2. exécuter le script de validation concerné ;
3. corriger les erreurs ;
4. documenter les éventuelles limitations ;
5. seulement ensuite continuer.
```

## Règle 2 — Pas de téléchargement silencieux sur le disque interne

Avant tout chargement Hugging Face / Torch :

```python
configure_external_model_caches()
```

doit avoir été exécuté.

## Règle 3 — Pas de dépendance ComfyUI

Ne pas importer de package ComfyUI.

Il est permis d’étudier son comportement, mais le projet doit rester autonome.

## Règle 4 — Préférer Diffusers + PyTorch

Réutiliser les composants standards lorsque possible.

Éviter de dupliquer du code de scheduler, VAE ou text encoding sauf nécessité spécifique à PuLID.

## Règle 5 — Garder PuLID isolé

Toute logique PuLID doit être confinée dans une couche dédiée afin de pouvoir être remplacée plus tard.

## Règle 6 — Aucun modèle lourd dans Git

Ajouter dans `.gitignore` :

```gitignore
*.safetensors
*.ckpt
*.onnx
*.bin
*.pt
*.pth
outputs/*
!outputs/.gitkeep
cache/identity/*
!cache/identity/.gitkeep
```

## Règle 7 — Ne jamais modifier les fichiers du SSD

Les modèles sous :

```text
/Volumes/SSD/Documents/PuLID_models
```

doivent être considérés comme read-only, à l’exception des dossiers explicitement réservés aux caches automatiques.

---

# 22. Ordre de réalisation recommandé

Ordre strict conseillé :

```text
01. Bootstrap repo
02. Paths + configuration
03. Cache externe
04. Device MPS
05. Inspection modèles
06. InsightFace seul
07. Cache identité
08. SDXL seul
09. Gestion mémoire
10. Étude intégration officielle PuLID
11. PuLID adapter
12. PuLID + SDXL
13. Générateur haut niveau
14. CLI
15. Benchmark
16. Tests supplémentaires
17. README
18. Compatibilité CUDA
```

Ne pas commencer l’étape 10 avant que les étapes 1 à 8 soient fonctionnelles.

---

# 23. Premier milestone

Le premier milestone considéré comme réussi est :

```bash
python scripts/test_insightface.py --image inputs/reference.jpg
```

puis :

```bash
python scripts/test_sdxl.py \
  --prompt "professional portrait photo" \
  --seed 42
```

Ces deux commandes doivent fonctionner indépendamment.

---

# 24. Deuxième milestone

Le deuxième milestone est :

```bash
python scripts/test_pulid.py \
  --reference inputs/reference.jpg \
  --prompt "cinematic portrait, realistic photography" \
  --seed 42
```

avec :

```text
✓ identité reconnaissable
✓ fichier PNG dans ./outputs
✓ metadata JSON
✓ aucun modèle écrit dans le repo
✓ aucun cache lourd écrit sur le disque interne
✓ génération MPS fonctionnelle
```

---

# 25. Architecture finale souhaitée

L’objectif à terme est de pouvoir écrire :

```python
from pulid_app import ImageGenerator

generator = ImageGenerator.from_config("config/default.yaml")

identity = generator.identity(
    "inputs/character.jpg",
    cache_key="character-001",
)

result = generator.generate(
    prompt="cinematic photograph of a man driving a vintage sports car",
    identity=identity,
    identity_strength=0.8,
    seed=42,
)

print(result.image_path)
```

avec un résultat similaire à :

```text
./outputs/2026-08-13_180000_character-001_42.png
```

---

# 26. Extension future

Après stabilisation de PuLID, introduire une interface commune :

```python
class IdentityAdapter(Protocol):
    def load(self): ...
    def encode(self, images): ...
    def condition(self, pipeline, identity, strength): ...
    def unload(self): ...
```

Implémentations futures :

```text
PuLIDAdapter
PhotoMakerAdapter
InstantIDAdapter
FaceSnapAdapter
```

Cela permettra de benchmarker plusieurs stratégies d’identité sans modifier le reste du pipeline.

---

# Définition de “terminé”

Le projet est considéré fonctionnel lorsque :

- [ ] le SSD externe est l’unique emplacement des poids lourds ;
- [ ] le checkpoint SDXL local fonctionne sous MPS ;
- [ ] AntelopeV2 détecte et encode une face ;
- [ ] PuLID v1.1 conditionne réellement SDXL ;
- [ ] une identité peut être mise en cache ;
- [ ] plusieurs générations peuvent réutiliser cette identité ;
- [ ] les images sont écrites dans `./outputs` ;
- [ ] les métadonnées sont enregistrées avec chaque génération ;
- [ ] le projet fonctionne sans ComfyUI ;
- [ ] un CLI permet de lancer une génération ;
- [ ] les composants sont suffisamment découplés pour ajouter un autre identity adapter ;
- [ ] l’architecture reste compatible avec un futur backend CUDA.
