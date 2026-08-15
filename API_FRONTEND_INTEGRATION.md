# API PuLID — intégration frontend

Ce serveur HTTP local expose exactement deux routes applicatives :

- `GET /models` liste séparément les checkpoints SDXL, les méthodes de sampling
  et les courbes de sigmas compatibles ;
- `POST /generate` génère une image et renvoie directement le PNG.

La génération HTTP est séparée du chemin CLI avec sauvegarde. Elle ne crée ni
image dans `outputs/`, ni manifeste JSON. L'image de référence, le
conditionnement PuLID et le PNG de réponse restent en mémoire. Le petit embedding
ArcFace est créé ou réutilisé sous `cache/identity/`, avec une clé dérivée des
pixels décodés et du nom du personnage. Le frontend reste responsable du
téléchargement ou du stockage du PNG reçu.

## Démarrage du serveur

Installer les dépendances du serveur avec celles de l'inférence :

```bash
uv pip install -e '.[inference,pulid,server]'
```

Démarrer sur l'interface locale :

```bash
pulid-server --host 127.0.0.1 --port 12693 --device mps
```

Si le frontend est servi depuis une autre origin, l'autoriser explicitement :

```bash
pulid-server \
  --host 127.0.0.1 \
  --port 12693 \
  --device mps \
  --cors-origin http://localhost:3000
```

`--cors-origin` est répétable. Par défaut, aucune origin distante n'est
autorisée. Le serveur n'implémente pas d'authentification et doit rester lié à
`127.0.0.1`, sauf si une protection réseau adaptée est ajoutée.

URL de base utilisée dans les exemples :

```text
http://127.0.0.1:12693
```

## 1. Lister les modèles, méthodes et sigmas

### Requête

```http
GET /models
```

```bash
curl http://127.0.0.1:12693/models
```

### Réponse `200 application/json`

```json
{
  "models": [
    {
      "name": "realvisxlV50_v50LightningBakedvae",
      "filename": "realvisxlV50_v50LightningBakedvae.safetensors",
      "default": true
    },
    {
      "name": "reaxl_v30",
      "filename": "reaxl_v30.safetensors",
      "default": false
    }
  ],
  "sampling_methods": [
    {
      "name": "default",
      "label": "Scheduler du checkpoint",
      "default": true,
      "supported_sigma_schedules": ["normal"]
    },
    {
      "name": "dpmpp_2m",
      "label": "DPM++ 2M",
      "default": false,
      "supported_sigma_schedules": ["normal", "karras", "exponential", "beta"]
    },
    {
      "name": "dpmpp_2m_sde",
      "label": "DPM++ 2M SDE",
      "default": false,
      "supported_sigma_schedules": ["normal", "karras", "exponential", "beta"]
    },
    {
      "name": "dpmpp_3m_sde",
      "label": "DPM++ 3M SDE",
      "default": false,
      "supported_sigma_schedules": ["normal", "karras", "exponential", "beta"]
    },
    {
      "name": "euler",
      "label": "Euler",
      "default": false,
      "supported_sigma_schedules": ["normal", "karras", "exponential", "beta"]
    },
    {
      "name": "euler_ancestral",
      "label": "Euler ancestral",
      "default": false,
      "supported_sigma_schedules": ["normal"]
    },
    {
      "name": "heun",
      "label": "Heun",
      "default": false,
      "supported_sigma_schedules": ["normal", "karras", "exponential", "beta"]
    },
    {
      "name": "lms",
      "label": "LMS",
      "default": false,
      "supported_sigma_schedules": ["normal", "karras", "exponential", "beta"]
    },
    {
      "name": "ddim",
      "label": "DDIM",
      "default": false,
      "supported_sigma_schedules": ["normal"]
    }
  ],
  "sigma_schedules": [
    {
      "name": "normal",
      "label": "Normal / natif",
      "default": true,
      "supported_sampling_methods": [
        "default", "dpmpp_2m", "dpmpp_2m_sde", "dpmpp_3m_sde",
        "euler", "euler_ancestral", "heun", "lms", "ddim"
      ]
    },
    {
      "name": "karras",
      "label": "Karras",
      "default": false,
      "supported_sampling_methods": [
        "dpmpp_2m", "dpmpp_2m_sde", "dpmpp_3m_sde", "euler", "heun", "lms"
      ]
    },
    {
      "name": "exponential",
      "label": "Exponentiel",
      "default": false,
      "supported_sampling_methods": [
        "dpmpp_2m", "dpmpp_2m_sde", "dpmpp_3m_sde", "euler", "heun", "lms"
      ]
    },
    {
      "name": "beta",
      "label": "Beta",
      "default": false,
      "supported_sampling_methods": [
        "dpmpp_2m", "dpmpp_2m_sde", "dpmpp_3m_sde", "euler", "heun", "lms"
      ]
    }
  ]
}
```

Le frontend doit envoyer la propriété `name`, sans extension, dans le champ
`model` de `/generate`. Il doit envoyer séparément le `name` d'une méthode dans
`method` et le `name` d'une courbe dans `sigmas`.

La méthode `default` conserve le scheduler fourni par le checkpoint et accepte
uniquement `normal`. Pour alimenter la liste des sigmas, filtrer
`sigma_schedules` avec `samplingMethod.supported_sigma_schedules`. Le champ
inverse `sigmaSchedule.supported_sampling_methods` est fourni pour les interfaces
qui partent d'abord du choix de sigmas. L'API refuse une combinaison incompatible
en `422` avant de charger le pipeline.

Migration de l'ancienne valeur combinée :

```text
method=dpmpp_2m_sde_karras
            devient
method=dpmpp_2m_sde + sigmas=karras
```

`dpmpp_2m_sde_karras` n'est plus une valeur acceptée pour `method`.

Les modèles sont relus à chaque appel depuis :

```text
/Volumes/SSD/Documents/PuLID_models/checkpoints/*.safetensors
```

Un checkpoint ajouté dans ce dossier apparaît donc sans redémarrage du serveur.

## 2. Générer une image

### Requête

```http
POST /generate
Content-Type: multipart/form-data
```

Champs du formulaire :

| Champ | Type | Requis | Défaut | Description |
|---|---:|:---:|---:|---|
| `reference` | fichier | oui | — | JPEG, PNG, WebP, BMP ou TIFF, 20 Mio maximum |
| `character` | texte | oui | — | Nom du personnage, 1 à 100 caractères |
| `prompt` | texte | oui | — | Prompt positif, 1 à 4000 caractères et 255 jetons CLIP utiles maximum |
| `negative_prompt` | texte | non | prompt négatif du pipeline | Prompt négatif, 4000 caractères et 255 jetons CLIP utiles maximum ; une chaîne vide le désactive |
| `clip_skip_2` | booléen | non | `false` | Active « Clip skip 2 » selon la convention A1111 pour les deux encodeurs CLIP de SDXL |
| `model` | texte | oui | — | `name` renvoyé par `GET /models` |
| `cfg` | nombre | non | `7.0` | CFG entre 0 et 30 |
| `steps` | entier | non | `20` | Nombre de steps entre 1 et 200 |
| `strength` | nombre | non | `0.8` | Force d'identité PuLID, finie et supérieure ou égale à 0 |
| `method` | texte | non | `default` | `sampling_methods[].name` renvoyé par `GET /models` |
| `sigmas` | texte | non | `normal` | `sigma_schedules[].name` compatible avec `method` |
| `seed` | entier | non | `0` | `0` ou `-1` = aléatoire ; sinon 1 à 2^63−1 |

La résolution est actuellement fixée à `1024 × 1024`. Si `negative_prompt` est
omis, le prompt négatif par défaut du pipeline est appliqué. Envoyer une chaîne
vide permet de le désactiver explicitement. `clip_skip_2` accepte les valeurs
booléennes de formulaire reconnues par FastAPI, notamment `true` et `false`.
La valeur `true` est traduite en `clip_skip=1` pour Diffusers : elle sélectionne
`hidden_states[-3]`. La valeur `false` conserve le comportement SDXL natif,
`hidden_states[-2]`.

Le backend compte séparément les jetons produits par les deux tokenizers de
SDXL, sans inclure les marqueurs BOS/EOS. Jusqu'à 75 jetons utiles, l'encodage
natif Diffusers est conservé. De 76 à 255 jetons, le prompt est segmenté en
blocs CLIP puis leurs embeddings sont concaténés sans troncature. Si l'un des
deux tokenizers dépasse 255 jetons utiles, l'API répond en `422` avec
`PromptTooLongError`. Le nombre de jetons n'est pas équivalent au nombre de
mots ou de caractères ; le frontend peut donc conserver la limite de 4000
caractères et afficher le message précis renvoyé par l'API.

Exemple `curl` :

```bash
curl --fail-with-body \
  --output generation.png \
  --dump-header response-headers.txt \
  --form reference=@inputs/noemie.webp \
  --form character=noemie \
  --form 'prompt=cinematic portrait of a woman standing in Tokyo at night' \
  --form 'negative_prompt=bad anatomy, watermark' \
  --form clip_skip_2=true \
  --form model=reaxl_v30 \
  --form cfg=4.5 \
  --form steps=20 \
  --form strength=1.25 \
  --form method=dpmpp_2m_sde \
  --form sigmas=karras \
  --form seed=0 \
  http://127.0.0.1:12693/generate
```

### Réponse réussie

Le corps de la réponse est directement le fichier PNG :

```http
HTTP/1.1 200 OK
Content-Type: image/png
Content-Disposition: attachment; filename="noemie_20260813T200947_123456Z.png"
Cache-Control: no-store
X-Generation-Seed: 987654321
X-SDXL-Model: reaxl_v30
X-Sampling-Method: dpmpp_2m_sde
X-Sigma-Schedule: karras
```

Le nom est composé du personnage normalisé et de la date/heure UTC :

```text
<personnage>_YYYYMMDDTHHMMSS_microsecondesZ.png
```

Pour une seed `0` ou `-1`, `X-Generation-Seed` contient la seed aléatoire
réellement utilisée. Le frontend doit la conserver s'il veut permettre de
reproduire la génération.

### Exemple JavaScript/TypeScript

```ts
type GenerateInput = {
  reference: File;
  character: string;
  prompt: string;
  negativePrompt?: string;
  clipSkip2?: boolean;
  model: string;
  cfg: number;
  steps: number;
  strength: number;
  method: string;
  sigmas: string;
  seed: number;
};

export async function generateImage(input: GenerateInput) {
  const form = new FormData();
  form.append("reference", input.reference);
  form.append("character", input.character);
  form.append("prompt", input.prompt);
  if (input.negativePrompt !== undefined) {
    form.append("negative_prompt", input.negativePrompt);
  }
  form.append("clip_skip_2", String(input.clipSkip2 ?? false));
  form.append("model", input.model);
  form.append("cfg", String(input.cfg));
  form.append("steps", String(input.steps));
  form.append("strength", String(input.strength));
  form.append("method", input.method);
  form.append("sigmas", input.sigmas);
  form.append("seed", String(input.seed));

  const response = await fetch("http://127.0.0.1:12693/generate", {
    method: "POST",
    body: form,
  });

  if (!response.ok) {
    const error = await response.json();
    throw new Error(
      error.detail?.message ?? `Génération impossible (${response.status})`,
    );
  }

  const blob = await response.blob();
  const disposition = response.headers.get("Content-Disposition") ?? "";
  const filename =
    disposition.match(/filename="([^"]+)"/)?.[1] ?? "generation.png";

  return {
    blob,
    objectUrl: URL.createObjectURL(blob),
    filename,
    seed: Number(response.headers.get("X-Generation-Seed")),
    model: response.headers.get("X-SDXL-Model"),
    method: response.headers.get("X-Sampling-Method"),
    sigmas: response.headers.get("X-Sigma-Schedule"),
  };
}
```

Pour le contrôle frontend, utiliser `0.8` comme valeur initiale et un champ
numérique avec `min=0`. Un pas de `0.05` ou `0.1` est adapté à l'interface, mais
l'API accepte toute valeur finie supérieure ou égale à zéro. `String(number)`
produit le séparateur décimal `.` attendu dans le `FormData`, indépendamment de
la langue de l'interface.

Types conseillés pour la réponse du GET :

```ts
type NamedOption = {
  name: string;
  label: string;
  default: boolean;
};

type SamplingMethod = NamedOption & {
  supported_sigma_schedules: string[];
};

type SigmaSchedule = NamedOption & {
  supported_sampling_methods: string[];
};

type ModelsResponse = {
  models: Array<{ name: string; filename: string; default: boolean }>;
  sampling_methods: SamplingMethod[];
  sigma_schedules: SigmaSchedule[];
};
```

Lorsqu'un utilisateur change de méthode, conserver les sigmas seulement s'ils
figurent encore dans `supported_sigma_schedules`; sinon revenir à la première
valeur compatible, normalement `normal` :

```ts
const method = inventory.sampling_methods.find((item) => item.name === methodName);
const allowed = new Set(method?.supported_sigma_schedules ?? ["normal"]);
const visibleSigmas = inventory.sigma_schedules.filter((item) => allowed.has(item.name));
const nextSigmas = allowed.has(currentSigmas) ? currentSigmas : visibleSigmas[0].name;
```

Lorsque l'aperçu n'est plus utilisé, libérer son URL :

```ts
URL.revokeObjectURL(result.objectUrl);
```

Pour télécharger le résultat côté navigateur :

```ts
const link = document.createElement("a");
link.href = result.objectUrl;
link.download = result.filename;
link.click();
```

## Erreurs

Les erreurs métier utilisent cette forme :

```json
{
  "detail": {
    "error": "ModelNotFoundError",
    "message": "Checkpoint SDXL introuvable: ..."
  }
}
```

Statuts usuels :

- `422` : formulaire invalide (dont `strength` négatif ou non fini), prompt
  dépassant 255 jetons CLIP utiles,
  modèle/méthode/sigmas inconnus, combinaison
  méthode-sigmas incompatible, image illisible, aucun visage ou plusieurs visages ;
- `500` : échec de chargement d'un modèle ou erreur pendant l'inférence.

Les erreurs de validation FastAPI utilisent un tableau standard dans `detail`.

## Cycle de vie et concurrence

- une seule génération est exécutée à la fois ; une requête concurrente attend
  la fin de la précédente afin de protéger la mémoire MPS/CUDA ;
- le checkpoint choisi est chargé localement avec les téléchargements désactivés ;
- l'embedding ArcFace est créé ou réutilisé dans le cache NPZ configuré ;
- le PNG est encodé dans un buffer mémoire puis renvoyé immédiatement ;
- les composants du pipeline sont fermés après la réponse ;
- aucun PNG ni manifeste JSON n'est créé par l'application serveur.

Une génération peut prendre plusieurs dizaines de secondes. Le proxy ou le
client HTTP du frontend doit utiliser un timeout adapté, par exemple deux à cinq
minutes selon le matériel et le nombre de steps.
