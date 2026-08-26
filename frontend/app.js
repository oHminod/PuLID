const MAX_REFERENCE_BYTES = 20 * 1024 * 1024;
const MAX_SEED = 9223372036854775807n;
const REQUEST_TIMEOUT_MS = 5 * 60 * 1000;

const state = {
  inventory: null,
  resultUrl: null,
  referenceUrl: null,
  generating: false,
};

const elements = {
  backendUrl: document.querySelector("#backendUrl"),
  reconnectButton: document.querySelector("#reconnectButton"),
  connectionState: document.querySelector("#connectionState"),
  connectionLabel: document.querySelector("#connectionLabel"),
  form: document.querySelector("#generationForm"),
  reference: document.querySelector("#reference"),
  referencePreview: document.querySelector("#referencePreview"),
  uploadZone: document.querySelector("#uploadZone"),
  uploadTitle: document.querySelector("#uploadTitle"),
  uploadHint: document.querySelector("#uploadHint"),
  prompt: document.querySelector("#prompt"),
  promptCount: document.querySelector("#promptCount"),
  negativeMode: document.querySelector("#negativeMode"),
  negativePrompt: document.querySelector("#negativePrompt"),
  negativePromptField: document.querySelector("#negativePromptField"),
  negativeCount: document.querySelector("#negativeCount"),
  model: document.querySelector("#model"),
  method: document.querySelector("#method"),
  sigmas: document.querySelector("#sigmas"),
  seed: document.querySelector("#seed"),
  generateButton: document.querySelector("#generateButton"),
  formError: document.querySelector("#formError"),
  emptyResult: document.querySelector("#emptyResult"),
  generatingResult: document.querySelector("#generatingResult"),
  generatedImage: document.querySelector("#generatedImage"),
  resultMeta: document.querySelector("#resultMeta"),
  resultSeed: document.querySelector("#resultSeed"),
  resultModel: document.querySelector("#resultModel"),
  resultMethod: document.querySelector("#resultMethod"),
  resultSigmas: document.querySelector("#resultSigmas"),
  downloadButton: document.querySelector("#downloadButton"),
};

function setConnection(status, label) {
  elements.connectionState.className = `connection-state ${status}`;
  elements.connectionLabel.textContent = label;
}

function setError(message = "") {
  elements.formError.textContent = message;
  elements.formError.hidden = !message;
}

function selectDefault(items) {
  return items.find((item) => item.default) ?? items[0] ?? null;
}

function fillSelect(select, items, selectedName = null) {
  select.replaceChildren();
  for (const item of items) {
    const option = document.createElement("option");
    option.value = item.name;
    option.textContent = item.label ?? item.filename ?? item.name;
    select.append(option);
  }
  const fallback = selectDefault(items)?.name ?? "";
  select.value = items.some((item) => item.name === selectedName) ? selectedName : fallback;
  select.disabled = items.length === 0;
}

function refreshSigmaOptions() {
  if (!state.inventory) return;
  const method = state.inventory.sampling_methods.find(
    (item) => item.name === elements.method.value,
  );
  const supported = new Set(method?.supported_sigma_schedules ?? ["normal"]);
  const visible = state.inventory.sigma_schedules.filter((item) => supported.has(item.name));
  fillSelect(elements.sigmas, visible, elements.sigmas.value);
}

async function loadInventory() {
  setError();
  setConnection("loading", "Connexion au backend…");
  elements.generateButton.disabled = true;
  elements.model.disabled = true;
  elements.method.disabled = true;
  elements.sigmas.disabled = true;

  try {
    const response = await fetch("/api/models", { cache: "no-store" });
    if (!response.ok) {
      throw new Error(`Inventaire indisponible (${response.status}).`);
    }
    const inventory = await response.json();
    if (!Array.isArray(inventory.models) || inventory.models.length === 0) {
      throw new Error("Aucun checkpoint SDXL n’est disponible dans le backend.");
    }
    state.inventory = inventory;
    fillSelect(elements.model, inventory.models);
    fillSelect(elements.method, inventory.sampling_methods);
    refreshSigmaOptions();
    elements.generateButton.disabled = false;
    setConnection("connected", `${inventory.models.length} modèle${inventory.models.length > 1 ? "s" : ""} disponible${inventory.models.length > 1 ? "s" : ""}`);
  } catch (error) {
    state.inventory = null;
    elements.model.replaceChildren(new Option("Backend indisponible", ""));
    setConnection("error", "Backend indisponible");
    setError(
      `${error.message} Vérifiez que le backend est lancé sur ${elements.backendUrl.value}.`,
    );
  }
}

function updateCount(input, output) {
  output.textContent = String(input.value.length);
}

function updateNegativePromptMode() {
  const custom = elements.negativeMode.value === "custom";
  elements.negativePromptField.hidden = !custom;
  elements.negativePrompt.disabled = !custom;
  if (custom) elements.negativePrompt.focus();
}

function updateReferencePreview() {
  const file = elements.reference.files?.[0];
  if (state.referenceUrl) URL.revokeObjectURL(state.referenceUrl);
  state.referenceUrl = null;
  elements.referencePreview.hidden = true;

  if (!file) {
    elements.uploadTitle.innerHTML = "Ajouter un portrait <span>*</span>";
    elements.uploadHint.textContent = "JPEG, PNG, WebP, BMP ou TIFF · 20 Mio max.";
    return;
  }

  elements.uploadTitle.textContent = file.name;
  elements.uploadHint.textContent = `${(file.size / 1024 / 1024).toFixed(2)} Mio · Cliquer pour remplacer`;
  state.referenceUrl = URL.createObjectURL(file);
  elements.referencePreview.src = state.referenceUrl;
  elements.referencePreview.hidden = false;
}

function validateReference() {
  const file = elements.reference.files?.[0];
  if (!file) {
    elements.reference.setCustomValidity("Ajoutez une image de référence.");
  } else if (file.size > MAX_REFERENCE_BYTES) {
    elements.reference.setCustomValidity("L’image de référence dépasse 20 Mio.");
  } else {
    elements.reference.setCustomValidity("");
  }
}

function validateSeed() {
  const value = elements.seed.value.trim();
  let message = "";
  if (!value) {
    elements.seed.setCustomValidity("");
    return;
  }
  if (!/^-?\d+$/.test(value)) {
    message = "La seed doit être un entier.";
  } else {
    const seed = BigInt(value);
    if (seed < -1n || seed > MAX_SEED) {
      message = `La seed doit valoir -1, 0, ou être comprise entre 1 et ${MAX_SEED}.`;
    }
  }
  elements.seed.setCustomValidity(message);
}

function buildGenerationBody() {
  const form = new FormData();
  form.append("reference", elements.reference.files[0]);
  form.append("character", document.querySelector("#character").value.trim());
  form.append("prompt", elements.prompt.value.trim());

  if (elements.negativeMode.value === "custom") {
    form.append("negative_prompt", elements.negativePrompt.value);
  } else if (elements.negativeMode.value === "disabled") {
    form.append("negative_prompt", "");
  }

  form.append("clip_skip_2", String(document.querySelector("#clipSkip2").checked));
  form.append("model", elements.model.value);
  const optionalValues = {
    cfg: document.querySelector("#cfg").value,
    steps: document.querySelector("#steps").value,
    strength: document.querySelector("#strength").value,
    seed: elements.seed.value.trim(),
  };
  for (const [name, value] of Object.entries(optionalValues)) {
    if (value !== "") form.append(name, value);
  }
  form.append("method", elements.method.value);
  form.append("sigmas", elements.sigmas.value);
  return form;
}

async function responseError(response) {
  try {
    const payload = await response.json();
    if (typeof payload.detail?.message === "string") return payload.detail.message;
    if (Array.isArray(payload.detail)) {
      return payload.detail.map((item) => item.msg).filter(Boolean).join(" · ");
    }
  } catch {
    // La réponse n'est pas du JSON exploitable.
  }
  return `Génération impossible (${response.status}).`;
}

function filenameFromResponse(response) {
  const disposition = response.headers.get("Content-Disposition") ?? "";
  return disposition.match(/filename="([^"]+)"/)?.[1] ?? "generation.png";
}

function showGenerating(active) {
  state.generating = active;
  elements.generateButton.disabled = active || !state.inventory;
  elements.generateButton.classList.toggle("loading", active);
  elements.generateButton.querySelector(".button-label").textContent = active
    ? "Génération en cours"
    : "Générer l’image";
  if (active) {
    elements.emptyResult.hidden = true;
    elements.generatedImage.hidden = true;
    elements.generatingResult.hidden = false;
  }
}

function showResult(response, blob) {
  if (state.resultUrl) URL.revokeObjectURL(state.resultUrl);
  state.resultUrl = URL.createObjectURL(blob);
  const filename = filenameFromResponse(response);
  elements.generatedImage.src = state.resultUrl;
  elements.generatedImage.hidden = false;
  elements.emptyResult.hidden = true;
  elements.generatingResult.hidden = true;
  elements.resultSeed.textContent = response.headers.get("X-Generation-Seed") ?? "—";
  elements.resultModel.textContent = response.headers.get("X-SDXL-Model") ?? elements.model.value;
  elements.resultMethod.textContent = response.headers.get("X-Sampling-Method") ?? elements.method.value;
  elements.resultSigmas.textContent = response.headers.get("X-Sigma-Schedule") ?? elements.sigmas.value;
  elements.resultMeta.hidden = false;
  elements.downloadButton.href = state.resultUrl;
  elements.downloadButton.download = filename;
  elements.downloadButton.hidden = false;
}

async function generate(event) {
  event.preventDefault();
  setError();
  validateReference();
  validateSeed();

  if (!elements.form.checkValidity()) {
    elements.form.reportValidity();
    return;
  }
  if (!state.inventory || state.generating) return;

  showGenerating(true);
  const controller = new AbortController();
  const timeout = window.setTimeout(() => controller.abort(), REQUEST_TIMEOUT_MS);
  try {
    const response = await fetch("/api/generate", {
      method: "POST",
      body: buildGenerationBody(),
      signal: controller.signal,
    });
    if (!response.ok) throw new Error(await responseError(response));
    showResult(response, await response.blob());
  } catch (error) {
    elements.generatingResult.hidden = true;
    if (!state.resultUrl) elements.emptyResult.hidden = false;
    else elements.generatedImage.hidden = false;
    setError(
      error.name === "AbortError"
        ? "Le backend n’a pas répondu après cinq minutes. La génération a été interrompue côté interface."
        : error.message,
    );
  } finally {
    window.clearTimeout(timeout);
    showGenerating(false);
  }
}

function useDroppedFile(event) {
  event.preventDefault();
  elements.uploadZone.classList.remove("dragging");
  const file = event.dataTransfer?.files?.[0];
  if (!file) return;
  const transfer = new DataTransfer();
  transfer.items.add(file);
  elements.reference.files = transfer.files;
  validateReference();
  updateReferencePreview();
}

elements.reconnectButton.addEventListener("click", loadInventory);
elements.backendUrl.addEventListener("keydown", (event) => {
  if (event.key === "Enter") {
    event.preventDefault();
    loadInventory();
  }
});
elements.form.addEventListener("submit", generate);
elements.method.addEventListener("change", refreshSigmaOptions);
elements.negativeMode.addEventListener("change", updateNegativePromptMode);
elements.reference.addEventListener("change", () => {
  validateReference();
  updateReferencePreview();
});
elements.seed.addEventListener("input", validateSeed);
elements.prompt.addEventListener("input", () => updateCount(elements.prompt, elements.promptCount));
elements.negativePrompt.addEventListener("input", () =>
  updateCount(elements.negativePrompt, elements.negativeCount),
);
elements.uploadZone.addEventListener("dragover", (event) => {
  event.preventDefault();
  elements.uploadZone.classList.add("dragging");
});
elements.uploadZone.addEventListener("dragleave", () =>
  elements.uploadZone.classList.remove("dragging"),
);
elements.uploadZone.addEventListener("drop", useDroppedFile);
window.addEventListener("beforeunload", () => {
  if (state.resultUrl) URL.revokeObjectURL(state.resultUrl);
  if (state.referenceUrl) URL.revokeObjectURL(state.referenceUrl);
});

updateNegativePromptMode();

async function initialize() {
  try {
    const response = await fetch("/frontend-config.json", { cache: "no-store" });
    if (response.ok) {
      const config = await response.json();
      elements.backendUrl.value = config.backend_url;
    }
  } catch {
    // La valeur par défaut reste affichée si la configuration est indisponible.
  }
  await loadInventory();
}

initialize();
