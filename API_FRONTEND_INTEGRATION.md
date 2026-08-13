# API PuLID — intégration frontend

Ce serveur HTTP local expose exactement deux routes applicatives :

- `GET /models` liste les checkpoints SDXL et les méthodes de sampling ;
- `POST /generate` génère une image et renvoie directement le PNG.

La génération HTTP est entièrement séparée du chemin CLI avec sauvegarde. Elle
ne crée ni image dans `outputs/`, ni manifeste JSON, ni cache d'identité NPZ.
L'image de référence, l'embedding, le conditionnement PuLID et le PNG de réponse
restent en mémoire. Le frontend est responsable du téléchargement ou du stockage
du PNG reçu.

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

## 1. Lister les modèles et méthodes

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
      "default": true
    },
    {
      "name": "dpmpp_2m_sde_karras",
      "label": "dpmpp_2m_sde_karras",
      "default": false
    }
  ]
}
```

Le frontend doit envoyer la propriété `name`, sans extension, dans le champ
`model` de `/generate`. La méthode `default` conserve le scheduler fourni par le
checkpoint.

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
| `prompt` | texte | oui | — | Prompt positif, 1 à 4000 caractères |
| `model` | texte | oui | — | `name` renvoyé par `GET /models` |
| `cfg` | nombre | non | `7.0` | CFG entre 0 et 30 |
| `steps` | entier | non | `20` | Nombre de steps entre 1 et 200 |
| `method` | texte | non | `default` | Méthode renvoyée par `GET /models` |
| `seed` | entier | non | `0` | `0` ou `-1` = aléatoire ; sinon 1 à 2^63−1 |

La résolution est actuellement fixée à `1024 × 1024` et la force PuLID à
`0.8`. Le prompt négatif par défaut du pipeline est appliqué.

Exemple `curl` :

```bash
curl --fail-with-body \
  --output generation.png \
  --dump-header response-headers.txt \
  --form reference=@inputs/noemie.webp \
  --form character=noemie \
  --form 'prompt=cinematic portrait of a woman standing in Tokyo at night' \
  --form model=reaxl_v30 \
  --form cfg=4.5 \
  --form steps=20 \
  --form method=dpmpp_2m_sde_karras \
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
X-Sampling-Method: dpmpp_2m_sde_karras
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
  model: string;
  cfg: number;
  steps: number;
  method: string;
  seed: number;
};

export async function generateImage(input: GenerateInput) {
  const form = new FormData();
  form.append("reference", input.reference);
  form.append("character", input.character);
  form.append("prompt", input.prompt);
  form.append("model", input.model);
  form.append("cfg", String(input.cfg));
  form.append("steps", String(input.steps));
  form.append("method", input.method);
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
  };
}
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

- `422` : formulaire invalide, modèle ou méthode inconnue, image illisible,
  aucun visage ou plusieurs visages ;
- `500` : échec de chargement d'un modèle ou erreur pendant l'inférence.

Les erreurs de validation FastAPI utilisent un tableau standard dans `detail`.

## Cycle de vie et concurrence

- une seule génération est exécutée à la fois ; une requête concurrente attend
  la fin de la précédente afin de protéger la mémoire MPS/CUDA ;
- le checkpoint choisi est chargé localement avec les téléchargements désactivés ;
- l'identité est calculée sans cache NPZ ;
- le PNG est encodé dans un buffer mémoire puis renvoyé immédiatement ;
- les composants du pipeline sont fermés après la réponse ;
- aucun PNG, JSON ou fichier d'identité n'est créé par l'application serveur.

Une génération peut prendre plusieurs dizaines de secondes. Le proxy ou le
client HTTP du frontend doit utiliser un timeout adapté, par exemple deux à cinq
minutes selon le matériel et le nombre de steps.
