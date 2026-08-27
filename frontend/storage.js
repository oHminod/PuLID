(function initializePuLIDStorage(root) {
  "use strict";

  const SETTINGS_KEY = "pulid-studio:settings:v1";
  const DATABASE_NAME = "pulid-studio";
  const DATABASE_VERSION = 2;
  const ARTIFACT_STORE = "artifacts";
  const LAST_REFERENCE_KEY = "last-reference";

  function createPuLIDStorage(backends = {}) {
    const localStorageBackend = backends.localStorage;
    const indexedDbBackend = backends.indexedDB;

    function loadSettings() {
      try {
        const raw = localStorageBackend?.getItem(SETTINGS_KEY);
        if (!raw) return null;
        const parsed = JSON.parse(raw);
        if (parsed?.version !== 1 || !parsed.values || typeof parsed.values !== "object") {
          return null;
        }
        return parsed.values;
      } catch {
        return null;
      }
    }

    function saveSettings(values) {
      try {
        localStorageBackend?.setItem(
          SETTINGS_KEY,
          JSON.stringify({ version: 1, values }),
        );
        return Boolean(localStorageBackend);
      } catch {
        return false;
      }
    }

    function openDatabase() {
      if (!indexedDbBackend) {
        return Promise.reject(new Error("IndexedDB n’est pas disponible."));
      }
      return new Promise((resolve, reject) => {
        const request = indexedDbBackend.open(DATABASE_NAME, DATABASE_VERSION);
        request.onupgradeneeded = (event) => {
          const database = request.result;
          let artifactStore;
          if (!database.objectStoreNames.contains(ARTIFACT_STORE)) {
            artifactStore = database.createObjectStore(ARTIFACT_STORE, { keyPath: "id" });
          } else {
            artifactStore = request.transaction.objectStore(ARTIFACT_STORE);
          }
          if (event.oldVersion < 2) {
            artifactStore.delete("last-result");
          }
        };
        request.onsuccess = () => resolve(request.result);
        request.onerror = () => reject(request.error ?? new Error("Ouverture IndexedDB impossible."));
        request.onblocked = () => reject(new Error("IndexedDB est bloqué par un autre onglet."));
      });
    }

    async function putArtifact(id, value) {
      const database = await openDatabase();
      try {
        await new Promise((resolve, reject) => {
          const transaction = database.transaction(ARTIFACT_STORE, "readwrite");
          transaction.objectStore(ARTIFACT_STORE).put({ id, ...value });
          transaction.oncomplete = () => resolve();
          transaction.onerror = () => reject(
            transaction.error ?? new Error("Écriture IndexedDB impossible."),
          );
          transaction.onabort = transaction.onerror;
        });
      } finally {
        database.close();
      }
    }

    async function getArtifact(id) {
      const database = await openDatabase();
      try {
        return await new Promise((resolve, reject) => {
          const request = database
            .transaction(ARTIFACT_STORE, "readonly")
            .objectStore(ARTIFACT_STORE)
            .get(id);
          request.onsuccess = () => resolve(request.result ?? null);
          request.onerror = () => reject(
            request.error ?? new Error("Lecture IndexedDB impossible."),
          );
        });
      } finally {
        database.close();
      }
    }

    async function deleteArtifact(id) {
      const database = await openDatabase();
      try {
        await new Promise((resolve, reject) => {
          const transaction = database.transaction(ARTIFACT_STORE, "readwrite");
          transaction.objectStore(ARTIFACT_STORE).delete(id);
          transaction.oncomplete = () => resolve();
          transaction.onerror = () => reject(
            transaction.error ?? new Error("Suppression IndexedDB impossible."),
          );
          transaction.onabort = transaction.onerror;
        });
      } finally {
        database.close();
      }
    }

    function saveReference(file) {
      return putArtifact(LAST_REFERENCE_KEY, {
        blob: file,
        filename: file.name,
        contentType: file.type,
        lastModified: file.lastModified,
        savedAt: new Date().toISOString(),
      });
    }

    function loadReference() {
      return getArtifact(LAST_REFERENCE_KEY);
    }

    function clearReference() {
      return deleteArtifact(LAST_REFERENCE_KEY);
    }

    async function clearAll() {
      try {
        localStorageBackend?.removeItem(SETTINGS_KEY);
      } catch {
        // Le nettoyage IndexedDB reste tenté si localStorage est indisponible.
      }
      if (!indexedDbBackend) return;
      await new Promise((resolve, reject) => {
        const request = indexedDbBackend.deleteDatabase(DATABASE_NAME);
        request.onsuccess = () => resolve();
        request.onerror = () => reject(
          request.error ?? new Error("Suppression IndexedDB impossible."),
        );
        request.onblocked = () => reject(
          new Error("Fermez les autres onglets PuLID avant d’effacer les données."),
        );
      });
    }

    return Object.freeze({
      loadSettings,
      saveSettings,
      saveReference,
      loadReference,
      clearReference,
      clearAll,
    });
  }

  let browserLocalStorage;
  let browserIndexedDb;
  try {
    browserLocalStorage = root.localStorage;
  } catch {
    browserLocalStorage = undefined;
  }
  try {
    browserIndexedDb = root.indexedDB;
  } catch {
    browserIndexedDb = undefined;
  }

  root.createPuLIDStorage = createPuLIDStorage;
  root.PuLIDStorage = createPuLIDStorage({
    localStorage: browserLocalStorage,
    indexedDB: browserIndexedDb,
  });
})(globalThis);
