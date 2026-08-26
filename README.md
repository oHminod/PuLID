# PuLID pour rp-bot

Ce projet fournit le service local de génération d’images de
[`rp-bot`](https://github.com/oHminod/rp-bot). Il transforme l’avatar d’un
personnage et le contexte d’un message en une image SDXL tout en préservant son
identité grâce à PuLID.

Le service peut également être utilisé sans `rp-bot` grâce à un frontend web
basique inclus dans le dépôt. Une CLI et une API HTTP sont disponibles pour les
usages avancés. L’ensemble fonctionne sans ComfyUI.

## Ce que fournit le projet

| Usage | Interface | Adresse par défaut |
|---|---|---|
| Génération depuis une conversation `rp-bot` | Action PuLID dans le chat | `http://127.0.0.1:12693` |
| Génération manuelle | Frontend web inclus | `http://127.0.0.1:8888` |
| Embeddings de texte pour la mémoire de `rp-bot` | API compatible OpenAI | `http://127.0.0.1:12693/v1` |
| Scripts et automatisations | CLI `pulid-gen` | terminal |

Le backend prend en charge Apple Silicon avec MPS, les GPU NVIDIA avec CUDA et
le CPU. Sur macOS, la détection faciale InsightFace reste exécutée sur CPU.

## Installation rapide

### Prérequis

- une connexion Internet lors de la première installation ;
- Python 3.11 à 3.13, installé automatiquement par les scripts si nécessaire ;
- sur macOS, un Mac Apple Silicon ;
- sous Windows, un GPU NVIDIA et un pilote compatible avec CUDA 13 ;
- au moins 20 Go disponibles, davantage si plusieurs checkpoints SDXL sont
  installés ;
- sous Windows, Microsoft C++ Build Tools si InsightFace doit être compilé.

Les modèles sont volumineux et ne sont pas versionnés avec le projet. Sur le Mac
prévu pour ce dépôt, utilisez le SSD externe :

```text
/Volumes/SSD/Documents/PuLID_models
```

L’installateur peut aussi utiliser un autre dossier `PuLID_models`. Il y place
les checkpoints, PuLID, AntelopeV2, BGE-M3 et tous les caches lourds.

### macOS

Depuis la racine du projet :

```bash
./install_macos.sh
```

À la première exécution, indiquez l’emplacement des modèles. Pour utiliser le
SSD externe, refusez l’emplacement proposé puis saisissez :

```text
/Volumes/SSD/Documents/PuLID_models
```

### Windows

Depuis l’Explorateur ou `cmd.exe` :

```bat
install_windows.bat
```

Le script installe l’environnement Python, les dépendances CUDA et les modèles,
puis tente de configurer le pare-feu Windows pour le port `12693` sur les
réseaux privés.

#### Si l’installation d’InsightFace échoue

Si l’installation s’arrête avec une erreur indiquant que Microsoft Visual C++
14 ou `cl.exe` est requis, ou qu’une wheel InsightFace ne peut pas être
compilée :

1. ouvrez la page officielle
   [Téléchargements Visual Studio](https://visualstudio.microsoft.com/downloads/) ;
2. dans **Outils pour Visual Studio**, téléchargez **Build Tools for Visual
   Studio** dans sa version stable actuelle ;
3. exécutez l’installateur en tant qu’administrateur ;
4. dans l’onglet **Charges de travail**, sélectionnez **Développement Desktop en
   C++** (`Desktop development with C++`) ;
5. conservez les composants recommandés et vérifiez que les outils MSVC pour
   x64/x86 ainsi qu’un SDK Windows récent sont sélectionnés ;
6. cliquez sur **Installer**, puis redémarrez Windows si l’installateur le
   demande ;
7. ouvrez un nouveau terminal dans le dossier PuLID et relancez :

   ```bat
   install_windows.bat
   ```

Si Build Tools est déjà présent, ouvrez **Visual Studio Installer**, choisissez
**Modifier**, puis ajoutez la charge de travail **Développement Desktop en
C++**. Le simple **Microsoft Visual C++ Redistributable** ne suffit pas : il
installe les bibliothèques d’exécution, mais pas le compilateur. La procédure
Microsoft détaillée est disponible dans
[Installer les outils Microsoft C++](https://learn.microsoft.com/cpp/overview/acquire-msvc?view=msvc-170).

### Choix du checkpoint SDXL

Si un checkpoint `.safetensors` existe déjà, placez-le dans le dossier
`checkpoints` de `PuLID_models` lorsque l’installateur le demande. Sinon, vous
pouvez accepter le téléchargement de SDXL Base 1.0.

L’installation est réparable et idempotente : relancer le script de votre
plateforme vérifie les fichiers présents et ne récupère que ce qui manque ou ce
qui est invalide.

## Utilisation avec rp-bot

`rp-bot` est l’interface principale prévue pour ce service. Il prépare le prompt
à partir du message et de la scène, envoie l’avatar de l’auteur comme référence,
puis conserve l’image générée dans la discussion.

### 1. Démarrer PuLID

Sur macOS :

```bash
./start_pulid_server.sh
```

Sur Windows :

```bat
start_windows.bat
```

Le serveur écoute sur le port `12693`. Le script Windows affiche aussi les
adresses IPv4 utilisables depuis une autre machine du réseau local.

Laissez ce terminal ouvert pendant l’utilisation de `rp-bot`. `Ctrl+C` arrête
le service.

### 2. Configurer le prompt SDXL dans rp-bot

Dans `rp-bot`, ouvrez **Réglages > Tâches LLM et prompts**, puis configurez la
tâche **Prompt d’image SDXL** avec un fournisseur et un modèle de texte actifs.
Cette tâche transforme le message de roleplay en prompt visuel adapté au
checkpoint sélectionné.

### 3. Connecter le service d’image

Ouvrez **Réglages > Modèles d’image**, puis la section **Génération SDXL avec
PuLID** :

1. activez la génération SDXL avec PuLID ;
2. renseignez l’URL du serveur ;
3. choisissez le checkpoint, la méthode de sampling et les autres paramètres ;
4. cliquez sur **Enregistrer**.

Utilisez l’une de ces adresses :

- `http://127.0.0.1:12693` si PuLID et `rp-bot` tournent sur la même machine ;
- `http://<IP_DU_PC>:12693` si PuLID tourne sur un PC Windows du réseau local.

Le badge **Catalogue disponible** confirme que `rp-bot` atteint le serveur. Le
bouton **Actualiser** recharge la liste des checkpoints et des samplers.

### 4. Générer depuis une conversation

Le personnage doit disposer d’un avatar contenant un visage exploitable. Dans
le chat, après un message du personnage :

1. survolez ou sélectionnez la bulle du message ;
2. cliquez sur l’action au visage intitulée **Générer avec PuLID** ;
3. attendez la fin de la génération.

Les réglages rapides **Génération SDXL PuLID** sont également disponibles dans
le panneau du chat. L’option **Inspecter le prompt** permet de relire et modifier
les prompts positif et négatif avant leur envoi.

L’image finale apparaît dans la galerie de la conversation. `rp-bot` la stocke
dans son propre dossier `.local-data/generated-images` avec le checkpoint, le
sampler, les sigmas et la seed réellement utilisés.

## Frontend autonome

Le frontend inclus permet de générer une image sans lancer `rp-bot`. Démarrez
d’abord le backend, puis ouvrez un second terminal.

Sur macOS, dans deux terminaux distincts :

```bash
# Terminal 1
./start_pulid_server.sh

# Terminal 2
./start_frontend_macos.sh
```

Sur Windows, dans deux terminaux distincts :

```bat
rem Terminal 1
start_windows.bat

rem Terminal 2
start_frontend_windows.bat
```

Ouvrez ensuite [http://localhost:8888](http://localhost:8888).

Le formulaire demande une image de référence, le nom du personnage, un prompt
et un checkpoint. Il permet aussi de régler le prompt négatif, Clip Skip 2, le
CFG, les steps, la force d’identité, le sampler, les sigmas et la seed.

Le PNG reste dans le navigateur jusqu’à son téléchargement. Contrairement à la
CLI, le frontend ne crée pas automatiquement de fichier dans `outputs/`.

Pour cibler un backend situé ailleurs :

```bash
./start_frontend_macos.sh --backend-url http://192.168.1.20:12693
```

L’équivalent Windows accepte la même option :

```bat
start_frontend_windows.bat --backend-url http://192.168.1.20:12693
```

## Embeddings de texte pour rp-bot

Le même serveur expose BGE-M3 au format OpenAI pour la mémoire vectorielle de
`rp-bot` :

- `GET /v1/models` liste le modèle `text-embedding-bge-m3` ;
- `POST /v1/embeddings` calcule des vecteurs de 1024 dimensions.

Si le fournisseur **LM Studio local** de `rp-bot` sert uniquement aux
embeddings, configurez-le ainsi dans **Réglages > Fournisseurs** :

```text
URL de base : http://127.0.0.1:12693/v1
Clé API : vide
Activé : oui
```

Puis configurez la tâche **Embeddings locaux** dans **Réglages > Tâches LLM et
prompts** :

```text
fournisseur : LM Studio local
modèle : text-embedding-bge-m3
format : embedding
activé : oui
```

Si LM Studio sert également au chat, ne remplacez pas son URL : le backend PuLID
n’expose pas `/v1/chat/completions`. La procédure pour séparer les fournisseurs
et reconstruire les index LanceDB est détaillée dans
[`RP_BOT_TEXT_EMBEDDING_INTEGRATION.md`](RP_BOT_TEXT_EMBEDDING_INTEGRATION.md).

## Vérifier l’installation

Les scripts de démarrage activent automatiquement l’environnement virtuel. Pour
utiliser les commandes directement sur macOS ou Linux :

```bash
source .venv/bin/activate
pulid-gen --version
pulid-gen doctor
pulid-gen inspect-models --show-cache-env --fail-on-internal-cache
```

Sous Windows :

```bat
.venv\Scripts\pulid-gen.exe --version
.venv\Scripts\pulid-gen.exe doctor
```

`doctor` contrôle les checkpoints, les modèles de visage, PuLID, BGE-M3, les
permissions, le device et les dépendances critiques sans lancer une génération
complète.

## Ajouter un checkpoint SDXL

Déposez le fichier `.safetensors` dans :

```text
<PuLID_models>/checkpoints/
```

Le VAE du checkpoint par défaut
`realvisxlV50_v50LightningBakedvae.safetensors` est intégré : aucun VAE externe
n’est nécessaire. Le serveur charge toujours un fichier local explicite et ne
télécharge jamais implicitement un modèle SDXL pendant une génération.

Après l’ajout, utilisez **Actualiser** dans `rp-bot` ou rechargez le frontend
autonome. Le nom du modèle est affiché sans l’extension `.safetensors`.

## Génération en ligne de commande

La CLI est utile pour tester le pipeline indépendamment des interfaces web :

```bash
pulid-gen generate \
  --reference inputs/noemie.webp \
  --character noemie \
  --prompt "cinematic portrait of a woman standing in Tokyo at night" \
  --method dpmpp_2m_sde \
  --sigmas karras \
  --cfg 4.5 \
  --strength 0.8 \
  --steps 20 \
  --seed 42
```

La commande écrit le PNG et un manifeste JSON adjacent dans `outputs/`. Pour
sélectionner un autre checkpoint, ajoutez `--model NOM_DU_FICHIER` sans
`.safetensors`.

## Mémoire GPU et performances

Le serveur garde le pipeline chargé entre les requêtes CUDA utilisant le même
checkpoint. Les modes suivants permettent d’arbitrer entre vitesse et mémoire,
notamment lorsque SDXL et BGE-M3 partagent le GPU :

| Démarrage | Comportement |
|---|---|
| sans option | BGE-M3 et SDXL utilisent le GPU sans offload ; calculs CUDA concurrents |
| `--serialized-cuda` | conserve les modèles sur le GPU mais sérialise leurs calculs |
| `--partial` | déplace les encodeurs CLIP et le VAE SDXL sur CPU pendant un embedding |
| `--full` | décharge tout le pipeline SDXL pendant un embedding |
| `--CPU` | exécute BGE-M3 sur CPU et laisse SDXL sur le GPU |

Exemples :

```bash
./start_pulid_server.sh --partial
```

```bat
start_windows.bat --serialized-cuda
start_windows.bat --CPU
```

Si le mode par défaut provoque une erreur de mémoire CUDA, essayez d’abord
`--serialized-cuda`, puis `--partial` ou `--full`.

## Fichiers et stockage

```text
PuLID/
├── outputs/                 # images et JSON créés par la CLI
├── cache/identity/          # petits caches d’identité ArcFace
└── config/local.yaml        # configuration locale générée, ignorée par Git

<PuLID_models>/
├── checkpoints/            # checkpoints SDXL locaux
├── antelopev2/             # modèles InsightFace
├── text_embedding/         # BGE-M3 au format GGUF
├── sources/PuLID/          # code officiel épinglé
├── huggingface/            # cache Hugging Face externe
├── torch/                  # cache PyTorch externe
└── other/                  # autres caches lourds
```

Le backend HTTP conserve le PNG en mémoire et le renvoie au client. Il ne crée
pas d’image ni de manifeste dans `outputs/`. Le frontend ou `rp-bot` est
responsable de l’enregistrement du PNG reçu.

## Dépannage

| Symptôme | Action recommandée |
|---|---|
| Catalogue PuLID indisponible dans `rp-bot` | Vérifier que le serveur est démarré et que l’URL se termine par `:12693`, sans `/v1` |
| Serveur distant inaccessible | Utiliser l’IPv4 privée affichée par `start_windows.bat` et vérifier le profil privé du pare-feu |
| Aucun visage détecté | Choisir un avatar net, de face et suffisamment grand |
| Plusieurs visages détectés | Recadrer l’avatar afin qu’un seul visage soit visible |
| Checkpoint introuvable | Vérifier le fichier sous `<PuLID_models>/checkpoints/`, puis actualiser le catalogue |
| Erreur de mémoire CUDA | Redémarrer avec `--serialized-cuda`, `--partial` ou `--full` |
| `llama.dll` ou erreur Windows `0xc000001d` | Mettre à jour le pilote NVIDIA puis relancer `install_windows.bat` |
| Installation incomplète | Relancer l’installateur, puis exécuter `pulid-gen doctor` |

Le serveur ne possède ni authentification ni gestion d’utilisateurs. Ne
l’exposez pas à Internet ; limitez son accès à la machine locale ou à un réseau
privé de confiance.

## Documentation avancée

- [`API_FRONTEND_INTEGRATION.md`](API_FRONTEND_INTEGRATION.md) : contrat HTTP,
  paramètres de génération et exemples d’intégration ;
- [`RP_BOT_TEXT_EMBEDDING_INTEGRATION.md`](RP_BOT_TEXT_EMBEDDING_INTEGRATION.md) :
  configuration détaillée de BGE-M3 dans `rp-bot` ;
- [`PULID_CODEX_IMPLEMENTATION_PLAN.md`](PULID_CODEX_IMPLEMENTATION_PLAN.md) :
  architecture, phases d’implémentation et validation technique.

Pour exécuter les tests unitaires, sans réseau ni modèle lourd :

```bash
pytest -m unit
```

AntelopeV2 est distribué pour la recherche non commerciale. Consultez la
licence InsightFace avant tout autre usage.
