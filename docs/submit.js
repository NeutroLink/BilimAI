import {adoptSession, queuePlace, refusalFrom, sessionHeaders} from "./pilot-client.js";

(() => {
  "use strict";

  // Pilot gateway contract:
  // The session identity this page carries, the header it travels in and the shape of the gateway's
  // refusals all live in ./pilot-client.js, which the English page imports too — the two locales
  // differ in their wording, never in the protocol (local://public-pilot-contracts.md).
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
  const progressFill = document.querySelector(".progress-fill");
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
    waitStartedAt: 0,
    waitTimer: 0,
  };

  // The Russian page (docs/index.html) and the English one (docs/en/index.html) load this one
  // module, so the wording follows the page's own `lang`. Everything the form says outside these
  // live states is already written out in the markup of each page — including the first line the
  // waiting panel shows, which is read back off the page below.
  const english = document.documentElement.lang === "en";

  const statusLabels = english
    ? {
        queued: "The check will start in a moment",
        starting: "Getting your check ready",
        evaluating: "Reading the page and comparing it with the teacher's text",
        rendering: "Preparing the marked page and the PDF",
      }
    : {
        queued: "Проверка начнётся с минуты на минуту",
        starting: "Готовимся проверять вашу работу",
        evaluating: "Считываем страницу и сверяем её с текстом учителя",
        rendering: "Готовим помеченную страницу и PDF",
      };

  const pendingPhase = english ? "The check is still running" : "Проверка продолжается";

  // Every live string this module writes onto the screen, in the language of the page that loaded
  // it. Nothing here names the machinery behind the pilot: a teacher waiting for her page gets a
  // sentence about her page, and the words that describe the inside of the system stay out of the
  // user-visible text (2026-09-17).
  const ui = english
    ? {
        sending: "Sending your page…",
        waiting: "Your page is accepted. Please keep this window open: the first check of the day takes a few minutes.",
        done: "Done",
        nextPage: "Paste the teacher's text and upload the next page.",
        photoReady: "The photo is ready. Paste the teacher's text and confirm the settings.",
        languagePicked: "Russian model selected. Paste the teacher's text and upload a photo.",
        assignmentPicked: "Dictation selected. Paste the teacher's text and upload a photo.",
        noFile: "No file selected",
        fileHint: "One well-lit page with no cropped edges",
        wrongFormat: "Upload a photo in JPG, PNG or WEBP format.",
        tooBig: "The file is larger than 20 MB. Make the photo smaller and try again.",
        replaceHint: "click the page to replace it",
        uploadedWork: "Uploaded work",
        checkedWork: "Checked work",
        checkedAlt: "Checked work with BilimAI marks",
        kilobytes: "KB",
        megabytes: "MB",
        locale: "en-US",
        incompleteResult: "The result of the check could not be read.",
        noJobId: "The check was not given a number.",
        notConnected: "The check is not connected yet.",
        rejected: "The request could not be accepted",
        unreadableReply: "The reply could not be read.",
        checkFailed: "The check could not be completed.",
        checkComplete: "Check complete. Confirm every mark before using it.",
        resetFailed: "Could not reset the check.",
        resultFailed: "Could not get the check result.",
        resultHeadline: "Check complete",
        resultSummary: "The results are prepared for teacher review.",
        noMarks: "No marks found",
        noMarksDetail: "Review the returned page before confirming.",
        deviation: "Deviation found",
        expiryUnknown: "The result will be deleted in one hour.",
        expiryUntil: "The result is available until",
      }
    : {
        sending: "Отправляем вашу страницу…",
        waiting: "Работа принята. Пожалуйста, не закрывайте это окно: первая проверка за день занимает несколько минут.",
        done: "Готово",
        nextPage: "Вставьте текст учителя и загрузите следующую страницу.",
        photoReady: "Фотография готова. Вставьте текст учителя и подтвердите параметры.",
        languagePicked: "Русская модель выбрана. Вставьте текст учителя и загрузите фотографию.",
        assignmentPicked: "Диктант выбран. Вставьте текст учителя и загрузите фотографию.",
        noFile: "Файл не выбран",
        fileHint: "Одна хорошо освещённая страница без обрезанных краёв",
        wrongFormat: "Загрузите фотографию в формате JPG, PNG или WEBP.",
        tooBig: "Файл больше 20 МБ. Уменьшите фотографию и попробуйте снова.",
        replaceHint: "нажмите на страницу, чтобы заменить",
        uploadedWork: "Загруженная работа",
        checkedWork: "Проверенная работа",
        checkedAlt: "Проверенная работа с пометками BilimAI",
        kilobytes: "КБ",
        megabytes: "МБ",
        locale: "ru-RU",
        incompleteResult: "Не удалось прочитать результат проверки.",
        noJobId: "Не удалось получить номер проверки.",
        notConnected: "Проверка пока не подключена.",
        rejected: "Не удалось принять запрос",
        unreadableReply: "Не удалось прочитать ответ.",
        checkFailed: "Проверку не удалось завершить.",
        checkComplete: "Проверка завершена. Перед использованием подтвердите каждую пометку.",
        resetFailed: "Не удалось сбросить проверку.",
        resultFailed: "Не удалось получить результат проверки.",
        resultHeadline: "Проверка завершена",
        resultSummary: "Результаты подготовлены для проверки учителем.",
        noMarks: "Пометки не найдены",
        noMarksDetail: "Просмотрите возвращённую страницу перед подтверждением.",
        deviation: "Найдено отклонение",
        expiryUnknown: "Результат будет удалён через один час.",
        expiryUntil: "Результат доступен до",
      };

  // The panel's first line is the page's own sentence for the seconds the photo is on its way up,
  // read off the markup so both locales carry it in one place. A second check in the same page
  // reuses it instead of keeping whatever the first one ended on.
  const sendingPhase = progressPhase.textContent;

  function endpoint() {
    const configured = document.querySelector('meta[name="bilimai-api-url"]')?.content.trim() || "";
    const localOverride = ["127.0.0.1", "localhost"].includes(window.location.hostname)
      ? new URLSearchParams(window.location.search).get("api") || ""
      : "";
    const value = configured || localOverride;
    return value.endsWith("/") ? value.slice(0, -1) : value;
  }

  const apiBase = endpoint();

  function ordinal(place) {
    const teens = place % 100;
    if (teens >= 11 && teens <= 13) return `${place}th`;
    return `${place}${["th", "st", "nd", "rd"][place % 10] || "th"}`;
  }

  // «Вы 3-й в очереди» is the literal translation of the English label, but the pilot calls its own
  // line of waiting jobs by that same word, and it is one a teacher never has to read. She is told
  // instead how many works are still ahead of hers (2026-09-17).
  function worksWord(count) {
    const rest = count % 100;
    if (rest % 10 === 1 && rest !== 11) return "работа";
    if (rest % 10 >= 2 && rest % 10 <= 4 && (rest < 12 || rest > 14)) return "работы";
    return "работ";
  }

  function queueLabel(place) {
    if (english) return `You are ${ordinal(place)} in line`;
    const ahead = place - 1;
    if (ahead === 0) return "Следующая проверка — ваша";
    return `Перед вами ещё ${ahead} ${worksWord(ahead)}`;
  }

  // The phase says what the line is waiting for.
  function phaseText(payload) {
    const phase = statusLabels[payload.status] || pendingPhase;
    const place = queuePlace(payload);
    return place > 0 ? `${queueLabel(place)} · ${phase}` : phase;
  }

  // 429 is the one refusal the teacher can act on, so it is the one that says how long to wait. The
  // gateway's own sentence is not repeated: it names what ran out (its cards, its line of waiting
  // jobs), and a teacher reading this form is never told about any of that — the numbers are the
  // same ones the gateway reported, said as a wait (2026-09-17).
  function refusalMessage(refusal) {
    if (refusal.scope === "queue") {
      return english
        ? "Too many checks right now. Try again in a few minutes."
        : "Сейчас слишком много проверок. Попробуйте через несколько минут.";
    }
    if (refusal.retryAfter <= 0) {
      return english
        ? "The check could not be accepted. Please try again a little later."
        : "Не удалось принять работу. Попробуйте, пожалуйста, немного позже.";
    }
    const minutes = Math.max(1, Math.ceil(refusal.retryAfter / 60));
    return english
      ? `Too many checks from this device. Try again in ${minutes} minute${minutes === 1 ? "" : "s"}.`
      : `С этого устройства отправлено много работ. Попробуйте снова через ${minutes} мин.`;
  }

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
      preview.alt = `${ui.uploadedWork}: ${state.file.name}`;
      previewBadge.textContent = ui.uploadedWork;
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
    fileName.textContent = ui.noFile;
    fileDetail.textContent = ui.fileHint;
    setFeedback(ui.nextPage);
    updateSubmitState();
  }

  function formatSize(bytes) {
    if (bytes < 1024 * 1024) return `${Math.max(1, Math.round(bytes / 1024))} ${ui.kilobytes}`;
    return `${(bytes / (1024 * 1024)).toFixed(1)} ${ui.megabytes}`;
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
    fileName.textContent = ui.noFile;
    fileDetail.textContent = ui.fileHint;
    setFeedback(message, true);
    updateSubmitState();
  }

  function loadFile(file) {
    if (!file) return;
    if (!isAcceptedImage(file)) {
      rejectFile(ui.wrongFormat);
      return;
    }
    if (file.size > 20 * 1024 * 1024) {
      rejectFile(ui.tooBig);
      return;
    }

    resetResult();
    if (state.previewUrl) URL.revokeObjectURL(state.previewUrl);
    state.file = file;
    state.previewUrl = URL.createObjectURL(file);
    preview.src = state.previewUrl;
    preview.alt = `${ui.uploadedWork}: ${file.name}`;
    previewBadge.textContent = ui.uploadedWork;
    dropzone.classList.add("has-file");
    fileName.textContent = file.name;
    fileDetail.textContent = `${formatSize(file.size)} · ${ui.replaceHint}`;
    setFeedback(ui.photoReady);
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

  // The waiting bar. The first check of the day wakes a sleeping machine — the pilot measured that
  // at 10-15 minutes (2026-09-17), and the gateway reports nothing while it happens, so the bar is
  // time and not measurement: the fraction of 1 - e^(-t/7min) it shows is about 50% at five minutes
  // and 86% at fifteen. It is capped below 1 so a wait that outlives the estimate keeps creeping
  // instead of parking at a number, and the only way it reaches the end is the gateway saying the
  // check is complete — a bar that fills and then keeps waiting would be worse than no bar at all.
  const WAIT_TICK_MS = 250;                  // sub-pixel steps on a 440px card, for a quarter of the work of rAF
  const WAIT_TAU_MS = 7 * 60 * 1000;
  const WAIT_CEILING = 0.97;

  function paintWait() {
    const elapsed = Date.now() - state.waitStartedAt;
    progressFill.style.transform = `scaleX(${(WAIT_CEILING * (1 - Math.exp(-elapsed / WAIT_TAU_MS))).toFixed(4)})`;
  }

  // Called once the gateway has accepted the page: until then the panel names no progress it cannot
  // know about, which is why a refusal never gets to show a moving bar.
  function startWaiting() {
    state.waitStartedAt = Date.now();
    workspace.classList.add("is-waiting");
    paintWait();
    state.waitTimer = window.setInterval(paintWait, WAIT_TICK_MS);
  }

  function stopWaiting() {
    window.clearInterval(state.waitTimer);
    state.waitTimer = 0;
    workspace.classList.remove("is-waiting");
  }

  // The bar's last step, taken only on the gateway's word, with the beat its transition needs to be
  // seen before the result panel takes the overlay's place.
  async function finishWaiting() {
    window.clearInterval(state.waitTimer);
    state.waitTimer = 0;
    progressPhase.textContent = ui.done;
    progressFill.style.transform = "scaleX(1)";
    await new Promise((resolve) => window.setTimeout(resolve, 450));
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
    const title = String(finding.title || finding.word || finding.type || ui.deviation);
    const detail = String(finding.detail || finding.explanation || finding.message || "");
    return {title, detail};
  }

  function showResult(payload) {
    const evaluatedUrl = safeUrl(payload.evaluated_document_url, "image");
    const reportUrl = safeUrl(payload.report_pdf_url, "pdf");
    const publicUrl = safeUrl(payload.public_url, "page");
    const assessment = payload.assessment;
    if (!evaluatedUrl || !reportUrl || !assessment || typeof assessment !== "object") {
      throw new Error(ui.incompleteResult);
    }

    preview.src = evaluatedUrl;
    preview.alt = ui.checkedAlt;
    previewBadge.textContent = ui.checkedWork;
    workspace.classList.add("has-result");
    resultHeadline.textContent = String(assessment.headline || ui.resultHeadline);
    resultSummary.textContent = String(assessment.summary || ui.resultSummary);
    findingsList.replaceChildren();

    const findings = Array.isArray(assessment.findings) ? assessment.findings.map(normalizeFinding).filter(Boolean) : [];
    if (!findings.length) findings.push({title: ui.noMarks, detail: ui.noMarksDetail});
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
      ? ui.expiryUnknown
      : `${ui.expiryUntil} ${expires.toLocaleTimeString(ui.locale, {hour: "2-digit", minute: "2-digit"})}.`;
    result.hidden = false;
    result.focus({preventScroll: true});
    result.scrollIntoView({behavior: window.matchMedia("(prefers-reduced-motion: reduce)").matches ? "auto" : "smooth", block: "start"});
  }

  async function fetchJson(path, options = {}) {
    const response = await fetch(`${apiBase}${path}`, {
      ...options,
      headers: sessionHeaders(options.headers),
      credentials: "include",
    });
    const payload = await response.json().catch(() => null);
    // An identity is minted on a refusal as well as on an accepted job, so every response is read
    // for one before anything is decided about it.
    adoptSession(response, payload);
    if (!response.ok) {
      const refusal = refusalFrom(response, payload);
      if (refusal) throw new Error(refusalMessage(refusal));
      const message = payload?.error || payload?.detail || `${ui.rejected} (${response.status}).`;
      throw new Error(String(message));
    }
    if (response.status === 204) return null;  // Reset confirms with an empty body: DELETE /v1/submissions/{id}.
    if (!payload) throw new Error(ui.unreadableReply);
    return payload;
  }


  async function pollJob(jobId, generation) {
    while (generation === state.pollGeneration) {
      const payload = await fetchJson(`/v1/submissions/${encodeURIComponent(jobId)}`);
      if (payload.status === "complete") {
        await finishWaiting();
        showResult(payload);
        setFeedback(ui.checkComplete);
        return;
      }
      if (payload.status === "failed" || payload.status === "cancelled") {
        throw new Error(payload.error || ui.checkFailed);
      }
      progressPhase.textContent = phaseText(payload);
      await new Promise((resolve) => window.setTimeout(resolve, 1500));
    }
  }

  languageButtons.forEach((button) => {
    if (button.dataset.unavailable) return;
    button.addEventListener("click", () => {
      state.language = button.dataset.language;
      pressSingle(languageButtons, state.language, "data-language");
      setFeedback(ui.languagePicked);
      updateSubmitState();
    });
  });

  categoryButtons.forEach((button) => {
    if (button.dataset.unavailable) return;
    button.addEventListener("click", () => {
      state.assignment = button.dataset.assignment;
      pressSingle(categoryButtons, state.assignment, "data-assignment");
      setFeedback(ui.assignmentPicked);
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
      setFeedback(ui.notConnected, true);
      return;
    }

    resetResult();
    progressPhase.textContent = sendingPhase;
    setFeedback(ui.sending);
    setBusy(true);
    const body = new FormData();
    body.append("document", state.file, state.file.name);
    body.append("source_text", sourceText.value.trim());
    body.append("language", "ru");
    body.append("assignment_type", "dictation");

    try {
      const accepted = await fetchJson("/v1/submissions", {method: "POST", body});
      state.jobId = String(accepted.job_id || "");
      if (!state.jobId) throw new Error(ui.noJobId);
      const generation = ++state.pollGeneration;
      startWaiting();
      // The panel is what the teacher reads; the live region repeats its promise for whoever hears
      // the page instead of seeing it.
      setFeedback(ui.waiting);
      await pollJob(state.jobId, generation);
    } catch (error) {
      setFeedback(error instanceof Error ? error.message : ui.resultFailed, true);
    } finally {
      stopWaiting();
      setBusy(false);
    }
  });

  resetButton.addEventListener("click", async () => {
    resetButton.disabled = true;
    try {
      if (state.jobId) await fetchJson(`/v1/submissions/${encodeURIComponent(state.jobId)}`, {method: "DELETE"});
      clearSubmission();
    } catch (error) {
      setFeedback(error instanceof Error ? error.message : ui.resetFailed, true);
    } finally {
      resetButton.disabled = false;
    }
  });

  updateSubmitState();
})();
