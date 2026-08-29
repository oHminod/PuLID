# Intégrer les embeddings PuLID dans rp-bot

Le backend PuLID expose désormais le même sous-ensemble OpenAI que celui déjà
utilisé par l'adaptateur LM Studio de `rp-bot` :

- `GET /v1/models` pour le catalogue ;
- `POST /v1/embeddings` pour un texte ou un lot de textes.

Le modèle publié conserve l'identifiant `text-embedding-bge-m3` et produit des
vecteurs de dimension 1024.

## Option recommandée : réutiliser la configuration LM Studio

Cette option ne demande aucune modification de code dans `rp-bot` si sa
configuration LM Studio sert uniquement aux embeddings.

1. Sur le PC, vérifier ce fichier :

   ```text
   <PuLID>\PuLID_models\text_embedding\bge-m3-Q8_0.gguf
   ```

2. Après le `git pull`, exécuter `install_windows.bat`, puis choisir un mode de
   démarrage :

   ```bat
   start_windows.bat
   start_windows.bat --partial
   start_windows.bat --full
   start_windows.bat --CPU
   ```

   Sans option, BGE utilise CUDA sans offload SDXL. `--partial` déplace CLIP et
   le VAE SDXL sur CPU pendant BGE, `--full` décharge entièrement SDXL et
   `--CPU` conserve le comportement CPU sans offload.

   L'installation ajoute `.venv\Lib\site-packages\torch\lib` au chemin de
   recherche des DLL afin que `llama.dll` retrouve les bibliothèques CUDA 13
   déjà fournies avec PyTorch.

3. Depuis la machine qui exécute `rp-bot`, vérifier le catalogue en remplaçant
   `<IP_PC>` par l'adresse IPv4 affichée au démarrage de PuLID :

   ```bash
   curl http://<IP_PC>:12693/v1/models
   ```

4. Dans **Réglages > Fournisseurs** de `rp-bot`, modifier le fournisseur
   « LM Studio local » :

   ```text
   URL de base : http://<IP_PC>:12693/v1
   Clé API : vide
   Activé : oui
   ```

   Si les deux applications tournent sur le PC, utiliser plutôt
   `http://127.0.0.1:12693/v1`.

5. Tester la connexion, puis synchroniser le catalogue. Le modèle
   `text-embedding-bge-m3` doit apparaître.

6. Dans **Réglages > Tâches LLM**, affecter à `embedding_generation` :

   ```text
   fournisseur : LM Studio local
   modèle : text-embedding-bge-m3
   format : embedding
   activé : oui
   ```

Le fichier actuel `lib/ai/adapters/lmstudio.ts` délègue déjà les embeddings à
`postOpenAiEmbedding`. Celui-ci envoie exactement les champs `model` et `input`
attendus par PuLID et accepte la réponse `data[].index` / `data[].embedding`.

## Vérification directe

Avant de reconstruire les index, envoyer un appel minimal :

```bash
curl http://<IP_PC>:12693/v1/embeddings \
  --header "Content-Type: application/json" \
  --data '{"model":"text-embedding-bge-m3","input":"test de connexion"}'
```

Vérifier les points suivants :

- statut HTTP `200` ;
- `model` vaut `text-embedding-bge-m3` ;
- `data` contient un élément d'index `0` ;
- `data[0].embedding` contient 1024 nombres finis.

Le backend accepte jusqu'à 8192 tokens par texte avec la configuration par
défaut. Pour les encodeurs GGUF tels que BGE-M3, conserver `batch_size` au moins
égal à `context_size` dans `config/default.yaml`. Une valeur inférieure peut
provoquer une assertion native de llama.cpp sur les textes longs ; PuLID refuse
désormais cette configuration au démarrage.

Le contexte BGE reste fixé à 8192 tokens dans tous les modes. Avec `--CPU`,
`threads: 0` permet au traitement par lots d'utiliser les 20 CPU logiques d'un
i9 10 cœurs / 20 threads ; une valeur positive réintroduit une limite.

## Index LanceDB existants

Le modèle et sa dimension restent identiques, mais LM Studio et le runtime
`llama.cpp` embarqué peuvent utiliser des révisions différentes du moteur. Pour
éviter de mélanger dans un même index des vecteurs calculés par deux versions,
faire une reconstruction unique après la bascule :

```bash
npm run vector:rebuild
```

SQLite reste la source de vérité ; la commande reconstruit les index LanceDB
dérivés avec la nouvelle route d'embedding. Sauvegarder les données locales
avant toute maintenance reste conseillé.

## Si LM Studio sert aussi aux tâches de chat

Le backend PuLID n'expose pas `/v1/chat/completions`. Remplacer l'URL du seul
fournisseur LM Studio casserait donc les tâches de chat qui lui sont affectées.

Dans ce cas, conserver le fournisseur LM Studio existant et ajouter dans
`rp-bot` un fournisseur distinct, par exemple `pulid`, en suivant ces étapes :

1. ajouter `pulid` au type de fournisseur et à sa persistance dans
   `lib/db/repositories/providers.ts` ;
2. créer `lib/ai/adapters/pulid.ts` en réutilisant `fetchJson`,
   `postOpenAiEmbedding` et `normalizeLmStudioModels` ;
3. publier `listModels()` et `createEmbedding()`, et refuser explicitement les
   méthodes de chat puisque PuLID ne les sert pas ;
4. enregistrer la fabrique et le libellé dans
   `lib/settings/provider-model-settings.ts` ;
5. faire pointer le défaut de `embedding_generation` vers ce fournisseur dans
   `lib/settings/task-route-settings.ts` ;
6. adapter les formulaires, fixtures et tests de fournisseurs/tâches concernés.

Cette variante permet à LM Studio de conserver ses modèles de chat tout en
réservant PuLID aux embeddings et à SDXL.

## Concurrence et réseau

- PuLID sérialise les lots d'embeddings entre eux.
- Les embeddings et SDXL peuvent calculer simultanément sur CUDA par défaut.
  `--serialized-cuda`, `--partial`, `--full` et `--CPU` permettent d’adapter la
  concurrence et la mémoire au matériel.
- `start_windows.bat` lie l'API à `127.0.0.1:12693` par défaut. Le mode avancé
  `start_windows.bat --network` utilise explicitement `0.0.0.0` et CORS ouvert ;
  la règle de pare-feu privée n'est proposée que par
  `install_windows.bat --network`. Ne pas exposer ce port à Internet : l'API ne
  possède ni authentification ni limitation par utilisateur.
- Les appels fournisseur de `rp-bot` partent du serveur Next.js, donc le
  navigateur n'appelle pas directement l'IP du PC PuLID.
