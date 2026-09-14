(() => {
  "use strict";

  // Pilot gateway contract:
  // Submission authorization is enforced by the gateway's machine certificate, never by a browser-visible secret.
  // Result URLs are public, opaque and expire after one hour.
  const form = document.getElementById("submission-form");
  if (!form) return;

  const languageButtons = [...document.querySelectorAll("[data-language]")];
  const categoryButtons = [...form.querySelectorAll("[data-assignment]")];
  const fileInput = document.getElementById("assignment-file");
  const sourceText = document.getElementById("source-text");
  const dropzone = document.getElementById("upload-dropzone");
  const preview = document.getElementById("upload-preview");
  const previewBadge = document.getElementById("preview-badge");
  const fileName = document.getElementById("upload-file-name");
  const fileDetail = document.getElementById("upload-file-detail");
  const submitButton = document.getElementById("evaluate-assignment");
  const feedback = document.getElementById("submission-feedback");
  const workspace = document.getElementById("upload-workspace");
  const progressPhase = document.getElementById("evaluation-phase");
  const result = document.getElementById("assessment-result");
  const resultHeadline = document.getElementById("assessment-headline");
  const resultSummary = document.getElementById("assessment-summary");
  const findingsList = document.getElementById("assessment-findings");
  const reportLink = document.getElementById("download-report");
  const publicResultLink = document.getElementById("public-result-link");
  const resultExpiry = document.getElementById("result-expiry");
  const resetButton = document.getElementById("reset-evaluation");

  const state = {
    language: "",
    assignment: "",
    file: null,
    previewUrl: "",
    busy: false,
    jobId: "",
    pollGeneration: 0,
  };

  const statusLabels = {
    queued: "Запрос ждёт свободный GPU…",
    starting: "Запускаем модель на сервере…",
    evaluating: "Считываем страницу и сверяем её с текстом учителя…",
    rendering: "Готовим помеченную страницу и PDF…",
  };

  function endpoint() {
    const configured = document.querySelector('meta[name="bilimai-api-url"]')?.content.trim() || "";
    const localOverride = ["127.0.0.1", "localhost"].includes(window.location.hostname)
      ? new URLSearchParams(window.location.search).get("api") || ""
      : "";
    const value = configured || localOverride;
    return value.endsWith("/") ? value.slice(0, -1) : value;
  }

  const apiBase = endpoint();

  function setFeedback(message, isError = false) {
    feedback.textContent = message;
    feedback.classList.toggle("error", isError);
  }

  function updateSubmitState() {
    submitButton.disabled = state.busy
      || state.language !== "ru"
      || state.assignment !== "dictation"
      || !sourceText.value.trim()
      || !state.file;
  }

  function pressSingle(buttons, selected, attribute) {
    buttons.forEach((button) => {
      button.setAttribute("aria-pressed", String(button.getAttribute(attribute) === selected));
    });
  }

  function resetResult() {
    result.hidden = true;
    workspace.classList.remove("has-result");
    findingsList.replaceChildren();
    reportLink.removeAttribute("href");
    publicResultLink.removeAttribute("href");
    publicResultLink.hidden = true;
    resultExpiry.textContent = "";
    if (state.previewUrl && state.file) {
      preview.src = state.previewUrl;
      preview.alt = `Загруженная работа: ${state.file.name}`;
      previewBadge.textContent = "Загруженная работа";
    }
  }

  function clearSubmission() {
    state.pollGeneration += 1;
    state.jobId = "";
    resetResult();
    fileInput.value = "";
    sourceText.value = "";
    state.file = null;
    if (state.previewUrl) URL.revokeObjectURL(state.previewUrl);
    state.previewUrl = "";
    preview.removeAttribute("src");
    preview.alt = "";
    previewBadge.textContent = "";
    dropzone.classList.remove("has-file");
    fileName.textContent = "Файл не выбран";
    fileDetail.textContent = "Одна хорошо освещённая страница без обрезанных краёв";
    setFeedback("Вставьте текст учителя и загрузите следующую страницу.");
    updateSubmitState();
  }

  function formatSize(bytes) {
    if (bytes < 1024 * 1024) return `${Math.max(1, Math.round(bytes / 1024))} КБ`;
    return `${(bytes / (1024 * 1024)).toFixed(1)} МБ`;
  }

  function isAcceptedImage(file) {
    return ["image/jpeg", "image/png", "image/webp"].includes(file.type);
  }

  function rejectFile(message) {
    resetResult();
    fileInput.value = "";
    state.file = null;
    if (state.previewUrl) URL.revokeObjectURL(state.previewUrl);
    state.previewUrl = "";
    preview.removeAttribute("src");
    preview.alt = "";
    previewBadge.textContent = "";
    dropzone.classList.remove("has-file");
    fileName.textContent = "Файл не выбран";
    fileDetail.textContent = "Одна хорошо освещённая страница без обрезанных краёв";
    setFeedback(message, true);
    updateSubmitState();
  }

  function loadFile(file) {
    if (!file) return;
    if (!isAcceptedImage(file)) {
      rejectFile("Загрузите фотографию в формате JPG, PNG или WEBP.");
      return;
    }
    if (file.size > 20 * 1024 * 1024) {
      rejectFile("Файл больше 20 МБ. Уменьшите фотографию и попробуйте снова.");
      return;
    }

    resetResult();
    if (state.previewUrl) URL.revokeObjectURL(state.previewUrl);
    state.file = file;
    state.previewUrl = URL.createObjectURL(file);
    preview.src = state.previewUrl;
    preview.alt = `Загруженная работа: ${file.name}`;
    previewBadge.textContent = "Загруженная работа";
    dropzone.classList.add("has-file");
    fileName.textContent = file.name;
    fileDetail.textContent = `${formatSize(file.size)} · нажмите на страницу, чтобы заменить`;
    setFeedback("Фотография готова. Вставьте текст учителя и подтвердите параметры.");
    updateSubmitState();
  }

  function setBusy(busy) {
    state.busy = busy;
    workspace.classList.toggle("is-loading", busy);
    workspace.setAttribute("aria-busy", String(busy));
    fileInput.disabled = busy;
    sourceText.disabled = busy;
    languageButtons.forEach((button) => { button.disabled = busy || Boolean(button.dataset.unavailable); });
    categoryButtons.forEach((button) => { button.disabled = busy || Boolean(button.dataset.unavailable); });
    updateSubmitState();
  }

  function safeUrl(value, kind) {
    if (typeof value !== "string" || !value.trim()) return "";
    try {
      const url = new URL(value, window.location.href);
      if (["http:", "https:", "blob:"].includes(url.protocol)) return url.href;
      if (kind === "pdf" && url.protocol === "data:" && url.href.startsWith("data:application/pdf")) return url.href;
      if (kind === "image" && url.protocol === "data:" && url.href.startsWith("data:image/")) return url.href;
    } catch {
      return "";
    }
    return "";
  }

  function normalizeFinding(finding) {
    if (typeof finding === "string") return {title: finding, detail: ""};
    if (!finding || typeof finding !== "object") return null;
    const title = String(finding.title || finding.word || finding.type || "Найдено отклонение");
    const detail = String(finding.detail || finding.explanation || finding.message || "");
    return {title, detail};
  }

  function showResult(payload) {
    const evaluatedUrl = safeUrl(payload.evaluated_document_url, "image");
    const reportUrl = safeUrl(payload.report_pdf_url, "pdf");
    const publicUrl = safeUrl(payload.public_url, "page");
    const assessment = payload.assessment;
    if (!evaluatedUrl || !reportUrl || !assessment || typeof assessment !== "object") {
      throw new Error("Сервер вернул неполный результат проверки.");
    }

    preview.src = evaluatedUrl;
    preview.alt = "Проверенная работа с пометками BilimAI";
    previewBadge.textContent = "Проверенная работа";
    workspace.classList.add("has-result");
    resultHeadline.textContent = String(assessment.headline || "Проверка завершена");
    resultSummary.textContent = String(assessment.summary || "Результаты подготовлены для проверки учителем.");
    findingsList.replaceChildren();

    const findings = Array.isArray(assessment.findings) ? assessment.findings.map(normalizeFinding).filter(Boolean) : [];
    if (!findings.length) findings.push({title: "Пометки не найдены", detail: "Просмотрите возвращённую страницу перед подтверждением."});
    findings.forEach((finding) => {
      const item = document.createElement("li");
      const title = document.createElement("strong");
      const detail = document.createElement("span");
      title.textContent = finding.title;
      detail.textContent = finding.detail;
      item.append(title);
      if (finding.detail) item.append(detail);
      findingsList.append(item);
    });

    reportLink.href = reportUrl;
    reportLink.download = String(payload.report_filename || "bilimai-report.pdf");
    if (publicUrl) {
      publicResultLink.href = publicUrl;
      publicResultLink.hidden = false;
    }
    const expires = new Date(payload.expires_at);
    resultExpiry.textContent = Number.isNaN(expires.getTime())
      ? "Результат будет удалён через один час."
      : `Результат доступен до ${expires.toLocaleTimeString("ru-RU", {hour: "2-digit", minute: "2-digit"})}.`;
    result.hidden = false;
    result.focus({preventScroll: true});
    result.scrollIntoView({behavior: window.matchMedia("(prefers-reduced-motion: reduce)").matches ? "auto" : "smooth", block: "start"});
  }

  async function fetchJson(path, options = {}) {
    const response = await fetch(`${apiBase}${path}`, {...options, credentials: "include"});
    if (!response.ok) {
      const failure = await response.json().catch(() => null);
      const message = failure?.error || failure?.detail || `Сервер отклонил запрос (${response.status}).`;
      throw new Error(String(message));
    }
    if (response.status === 204) return null;  // Reset confirms with an empty body: DELETE /v1/submissions/{id}.
    const payload = await response.json().catch(() => null);
    if (!payload) throw new Error("Сервер вернул ответ, который не удалось прочитать.");
    return payload;
  }


  async function pollJob(jobId, generation) {
    while (generation === state.pollGeneration) {
      const payload = await fetchJson(`/v1/submissions/${encodeURIComponent(jobId)}`);
      if (payload.status === "complete") {
        showResult(payload);
        setFeedback("Проверка завершена. Перед использованием подтвердите каждую пометку.");
        return;
      }
      if (payload.status === "failed" || payload.status === "cancelled") {
        throw new Error(payload.error || "Проверку не удалось завершить.");
      }
      progressPhase.textContent = statusLabels[payload.status] || "Проверка продолжается…";
      await new Promise((resolve) => window.setTimeout(resolve, 1500));
    }
  }

  languageButtons.forEach((button) => {
    if (button.dataset.unavailable) return;
    button.addEventListener("click", () => {
      state.language = button.dataset.language;
      pressSingle(languageButtons, state.language, "data-language");
      setFeedback("Русская модель выбрана. Вставьте текст учителя и загрузите фотографию.");
      updateSubmitState();
    });
  });

  categoryButtons.forEach((button) => {
    if (button.dataset.unavailable) return;
    button.addEventListener("click", () => {
      state.assignment = button.dataset.assignment;
      pressSingle(categoryButtons, state.assignment, "data-assignment");
      setFeedback("Диктант выбран. Вставьте текст учителя и загрузите фотографию.");
      updateSubmitState();
    });
  });

  sourceText.addEventListener("input", updateSubmitState);
  fileInput.addEventListener("change", () => loadFile(fileInput.files?.[0]));
  dropzone.addEventListener("keydown", (event) => {
    if (event.key === "Enter" || event.key === " ") {
      event.preventDefault();
      fileInput.click();
    }
  });
  ["dragenter", "dragover"].forEach((eventName) => {
    dropzone.addEventListener(eventName, (event) => {
      event.preventDefault();
      dropzone.classList.add("is-dragging");
    });
  });
  ["dragleave", "drop"].forEach((eventName) => {
    dropzone.addEventListener(eventName, (event) => {
      event.preventDefault();
      dropzone.classList.remove("is-dragging");
    });
  });
  dropzone.addEventListener("drop", (event) => loadFile(event.dataTransfer?.files?.[0]));


  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    if (submitButton.disabled || state.busy) return;
    if (!apiBase) {
      setFeedback("Сервер пилотного теста пока не подключён.", true);
      return;
    }

    resetResult();
    setFeedback("Работа принята. Можно оставить эту вкладку открытой до завершения проверки.");
    setBusy(true);
    const body = new FormData();
    body.append("document", state.file, state.file.name);
    body.append("source_text", sourceText.value.trim());
    body.append("language", "ru");
    body.append("assignment_type", "dictation");

    try {
      const accepted = await fetchJson("/v1/submissions", {method: "POST", body});
      state.jobId = String(accepted.job_id || "");
      if (!state.jobId) throw new Error("Сервер не вернул номер задания.");
      const generation = ++state.pollGeneration;
      await pollJob(state.jobId, generation);
    } catch (error) {
      setFeedback(error instanceof Error ? error.message : "Не удалось получить результат проверки.", true);
    } finally {
      setBusy(false);
    }
  });

  resetButton.addEventListener("click", async () => {
    resetButton.disabled = true;
    try {
      if (state.jobId) await fetchJson(`/v1/submissions/${encodeURIComponent(state.jobId)}`, {method: "DELETE"});
      clearSubmission();
    } catch (error) {
      setFeedback(error instanceof Error ? error.message : "Не удалось сбросить проверку.", true);
    } finally {
      resetButton.disabled = false;
    }
  });

  updateSubmitState();
})();
