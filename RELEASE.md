# Publier PuLID

`pyproject.toml` est l’unique source de la version PuLID. Le nom de l’archive,
sa racine, `release-metadata.json` et les commandes de construction lisent tous
`project.version`; aucune version applicative n’est recopiée dans les scripts.

## Construire les artefacts

Depuis un clone ou depuis une précédente archive extraite :

```bash
python scripts/build_release.py
```

La commande crée :

```text
dist/pulid-<version>.tar.gz
dist/checksums-sha256.txt
```

L’archive est déterministe pour un même contenu : chemins triés, date à l’époque
Unix, propriétaires normalisés et en-tête gzip stable. Elle contient les sources,
le frontend, la configuration par défaut et les installateurs, mais exclut
notamment `.git`, `.github`, `tests`, les scripts `test_*`, environnements
virtuels, caches, `config/local.yaml`, entrées/sorties locales et fichiers de
modèles (`.safetensors`, `.onnx`, `.gguf`, `.pt`, `.pth`, `.ckpt`, `.bin`).

Vérifier l’empreinte sur macOS ou Linux :

```bash
cd dist
shasum -a 256 -c checksums-sha256.txt
```

Sur Windows PowerShell :

```powershell
$line = (Get-Content dist\checksums-sha256.txt).Split(' ', 2)
(Get-FileHash -Algorithm SHA256 (Join-Path dist $line[1].Trim())).Hash.ToLowerInvariant() -eq $line[0]
```

## Vérifier l’installation sans Git

Extraire l’archive dans un nouveau dossier, puis utiliser le profil production :

```bash
./install_production_macos.sh
```

ou sous Windows :

```bat
install_production_windows.bat
```

Ces wrappers réutilisent respectivement `install_macos.sh --production` et
`install_windows.bat --production`. Le profil production installe les extras
`inference`, `pulid`, `server` et `embeddings` sans installation éditable et sans
extra `dev`. Les modèles et `config/local.yaml` restent externes à l’archive ;
les installateurs réutilisent les fichiers valides déjà présents et ne les
téléchargent de nouveau que s’ils manquent ou échouent à leur contrôle
d’intégrité.

## Première GitHub Release

Avant une première release publique :

1. choisir et ajouter la licence du dépôt ;
2. vérifier l’état juridique de la redistribution du code, sans ajouter aucun
   modèle ou poids à l’archive ;
3. exécuter les tests unitaires et les validations documentées dans le README ;
4. reconstruire les artefacts sur un commit propre et vérifier le SHA-256 ;
5. signer l’archive et `checksums-sha256.txt` selon la politique de signature du
   canal stable ;
6. créer seulement ensuite le tag et la GitHub Release avec l’autorisation du
   propriétaire.

Pour préparer le nom du tag sans dupliquer la version :

```bash
VERSION="$(python -c 'import pathlib,tomllib; print(tomllib.loads(pathlib.Path("pyproject.toml").read_text())["project"]["version"])')"
echo "v${VERSION}"
```

La création du tag, le push et la publication ne font volontairement pas partie
de `build_release.py`.
