# AGENTS.md

## Mission du dépôt

Ce projet fournit un pipeline Python autonome pour générer des images avec SDXL et PuLID, d'abord sur Apple Silicon/MPS, puis sur CUDA. Le code doit rester modulaire afin qu'un autre adaptateur d'identité puisse remplacer PuLID sans réécrire le pipeline principal.

## Règles impératives

- Ne jamais télécharger ni copier de poids de modèles dans le dépôt.
- Conserver tous les modèles et caches volumineux sous `/Volumes/SSD/Documents/PuLID_models`.
- Configurer les variables de cache externes avant tout import de bibliothèque susceptible de télécharger des fichiers (`torch`, `transformers`, `diffusers`, `huggingface_hub`, `insightface`).
- Centraliser les chemins dans la configuration. Ne pas coder de chemin de checkpoint en dur dans les modules métier.
- Les checkpoints SDXL résident sous `/Volumes/SSD/Documents/PuLID_models/checkpoints`. Le checkpoint par défaut est `realvisxlV50_v50LightningBakedvae.safetensors`. Son VAE est intégré : ne pas charger ni exiger de VAE externe.
- Ne jamais déclencher un téléchargement implicite de SDXL. Le chargement doit viser un fichier local explicite.
- Écrire les générations et métadonnées dans `outputs/`, et les petits caches d'identité dans `cache/identity/`.
- Préserver la compatibilité MPS, CUDA et CPU. InsightFace/ONNX doit pouvoir rester sur CPU sur macOS.
- Lors de toute modification du serveur HTTP qui change le contrat d'un endpoint (route, méthode, paramètres, corps ou en-têtes de réponse, statuts ou erreurs), mettre à jour `API_FRONTEND_INTEGRATION.md` dans la même tranche de travail.

## Méthode de travail des agents

1. Lire `PULID_CODEX_IMPLEMENTATION_PLAN.md` et ce fichier avant une modification importante.
2. Inspecter l'état du dépôt et les changements existants ; ne pas écraser les modifications de l'utilisateur.
3. Implémenter la plus petite tranche complète répondant à la phase demandée.
4. Séparer la configuration, l'accès aux modèles, le pipeline et les entrées/sorties.
5. Fournir des erreurs explicites avec le chemin concerné et une action corrective.
6. Ajouter ou adapter des tests pour toute logique pure ou gestion de chemins.
7. Exécuter les tests pertinents et les scripts d'acceptation avant de conclure.
8. Ne pas poursuivre une phase ultérieure sans demande explicite.

## Style Python

- Cibler Python 3.11 ou supérieur et utiliser les annotations de type.
- Préférer `pathlib.Path`, les `dataclass` et des fonctions courtes à responsabilité unique.
- Garder les imports lourds au plus près de leur usage lorsqu'ils ne sont pas nécessaires au bootstrap.
- Éviter les états globaux, sauf la configuration idempotente des variables de cache.
- Ne pas masquer une exception utile ; enrichir le message tout en conservant la cause avec `raise ... from exc`.
- Les commandes CLI doivent retourner un code non nul en cas d'échec et produire une sortie lisible, concise et actionnable.

## Configuration et chemins

- Résoudre les chemins relatifs par rapport à la racine du projet, pas au répertoire courant du processus.
- Permettre les surcharges par variables d'environnement sans rendre la configuration YAML ambiguë.
- Valider séparément existence, type de fichier/dossier et permissions d'écriture.
- La détection de modèles peut être tolérante ; leur chargement doit être strict et explicite.

## Tests et validation

- Les tests unitaires ne doivent nécessiter ni réseau, ni SSD externe, ni modèle lourd.
- Utiliser des dossiers temporaires pour tester les chemins et la configuration.
- Marquer ou isoler les tests d'intégration qui requièrent MPS, CUDA ou des checkpoints.
- Pour la phase 1, valider au minimum `pytest` et `python scripts/inspect_models.py`.

## Git et artefacts

- Ne pas versionner `.venv`, checkpoints, sorties générées, logs, caches Python ou secrets.
- Conserver des `.gitkeep` uniquement pour les dossiers vides attendus par le projet.
- Faire des commits ciblés uniquement lorsque l'utilisateur le demande ; ne jamais publier ni pousser sans autorisation explicite.
