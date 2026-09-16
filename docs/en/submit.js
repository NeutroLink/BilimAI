import {adoptSession, queuePlace, refusalFrom, sessionHeaders} from "../pilot-client.js";

(() => {
  "use strict";

  // Pilot gateway contract:
  // The session identity this page carries, the header it travels in and the shape of the gateway's
  // refusals all live in ../pilot-client.js, which the Russian page imports too — the two locales
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
    queued: "Waiting for a free GPU…",
    starting: "Starting the model on the server…",
    evaluating: "Reading the page and cross-checking it against the teacher's text…",
    rendering: "Preparing the marked page and the PDF…",
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

  function ordinal(place) {
    const teens = place % 100;
    if (teens >= 11 && teens <= 13) return `${place}th`;
    return `${place}${["th", "st", "nd", "rd"][place % 10] || "th"}`;
  }

  // "You are 3rd in line" — the gateway reports a place only while a page is actually waiting, and
  // the phase says what the line is waiting for.
  function phaseText(payload) {
    const phase = statusLabels[payload.status] || "Check in progress…";
    const place = queuePlace(payload);
    return place > 0 ? `You are ${ordinal(place)} in line · ${phase}` : phase;
  }

  // The gateway answers in Russian only, so this page says the same thing in English from the same
  // numbers: how many minutes `retry_after` is, and which allowance ran out
  // (local://public-pilot-contracts.md §HTTP surface).
  function refusalMessage(refusal) {
    if (refusal.scope === "queue") return "The queue is full. Try again in a few minutes.";
    if (refusal.retryAfter <= 0) {
      return "The gateway refused the submission. Try again a little later.";
    }
    const minutes = Math.max(1, Math.ceil(refusal.retryAfter / 60));
    return `Too many checks from this device. Try again in ${minutes} minute${minutes === 1 ? "" : "s"}.`;
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
      preview.alt = `Uploaded work: ${state.file.name}`;
      previewBadge.textContent = "Uploaded work";
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
    fileName.textContent = "No file selected";
    fileDetail.textContent = "One well-lit page with no cropped edges";
    setFeedback("Paste the teacher's text and upload the next page.");
    updateSubmitState();
  }

  function formatSize(bytes) {
    if (bytes < 1024 * 1024) return `${Math.max(1, Math.round(bytes / 1024))} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
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
    fileName.textContent = "No file selected";
    fileDetail.textContent = "One well-lit page with no cropped edges";
    setFeedback(message, true);
    updateSubmitState();
  }

  function loadFile(file) {
    if (!file) return;
    if (!isAcceptedImage(file)) {
      rejectFile("Upload a photo in JPG, PNG or WEBP format.");
      return;
    }
    if (file.size > 20 * 1024 * 1024) {
      rejectFile("The file is larger than 20 MB. Make the photo smaller and try again.");
      return;
    }

    resetResult();
    if (state.previewUrl) URL.revokeObjectURL(state.previewUrl);
    state.file = file;
    state.previewUrl = URL.createObjectURL(file);
    preview.src = state.previewUrl;
    preview.alt = `Uploaded work: ${file.name}`;
    previewBadge.textContent = "Uploaded work";
    dropzone.classList.add("has-file");
    fileName.textContent = file.name;
    fileDetail.textContent = `${formatSize(file.size)} · click the page to replace it`;
    setFeedback("The photo is ready. Paste the teacher's text and confirm the settings.");
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
    const title = String(finding.title || finding.word || finding.type || "Deviation found");
    const detail = String(finding.detail || finding.explanation || finding.message || "");
    return {title, detail};
  }

  function showResult(payload) {
    const evaluatedUrl = safeUrl(payload.evaluated_document_url, "image");
    const reportUrl = safeUrl(payload.report_pdf_url, "pdf");
    const publicUrl = safeUrl(payload.public_url, "page");
    const assessment = payload.assessment;
    if (!evaluatedUrl || !reportUrl || !assessment || typeof assessment !== "object") {
      throw new Error("The server returned an incomplete check result.");
    }

    preview.src = evaluatedUrl;
    preview.alt = "Checked work with BilimAI marks";
    previewBadge.textContent = "Checked work";
    workspace.classList.add("has-result");
    resultHeadline.textContent = String(assessment.headline || "Check complete");
    resultSummary.textContent = String(assessment.summary || "The results are prepared for teacher review.");
    findingsList.replaceChildren();

    const findings = Array.isArray(assessment.findings) ? assessment.findings.map(normalizeFinding).filter(Boolean) : [];
    if (!findings.length) findings.push({title: "No marks found", detail: "Review the returned page before confirming."});
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
      ? "The result will be deleted in one hour."
      : `The result is available until ${expires.toLocaleTimeString("en-US", {hour: "2-digit", minute: "2-digit"})}.`;
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
      const message = payload?.error || payload?.detail || `The server rejected the request (${response.status}).`;
      throw new Error(String(message));
    }
    if (response.status === 204) return null;  // Reset confirms with an empty body: DELETE /v1/submissions/{id}.
    if (!payload) throw new Error("The server returned a response that could not be read.");
    return payload;
  }


  async function pollJob(jobId, generation) {
    while (generation === state.pollGeneration) {
      const payload = await fetchJson(`/v1/submissions/${encodeURIComponent(jobId)}`);
      if (payload.status === "complete") {
        showResult(payload);
        setFeedback("Check complete. Confirm every mark before using it.");
        return;
      }
      if (payload.status === "failed" || payload.status === "cancelled") {
        throw new Error(payload.error || "The check could not be completed.");
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
      setFeedback("Russian model selected. Paste the teacher's text and upload a photo.");
      updateSubmitState();
    });
  });

  categoryButtons.forEach((button) => {
    if (button.dataset.unavailable) return;
    button.addEventListener("click", () => {
      state.assignment = button.dataset.assignment;
      pressSingle(categoryButtons, state.assignment, "data-assignment");
      setFeedback("Dictation selected. Paste the teacher's text and upload a photo.");
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
      setFeedback("The pilot test server is not connected yet.", true);
      return;
    }

    resetResult();
    setFeedback("Work accepted. You can keep this tab open until the check completes.");
    setBusy(true);
    const body = new FormData();
    body.append("document", state.file, state.file.name);
    body.append("source_text", sourceText.value.trim());
    body.append("language", "ru");
    body.append("assignment_type", "dictation");

    try {
      const accepted = await fetchJson("/v1/submissions", {method: "POST", body});
      state.jobId = String(accepted.job_id || "");
      if (!state.jobId) throw new Error("The server did not return a job id.");
      const generation = ++state.pollGeneration;
      await pollJob(state.jobId, generation);
    } catch (error) {
      setFeedback(error instanceof Error ? error.message : "Could not get the check result.", true);
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
      setFeedback(error instanceof Error ? error.message : "Could not reset the check.", true);
    } finally {
      resetButton.disabled = false;
    }
  });

  updateSubmitState();
})();
