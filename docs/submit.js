import {adoptSession, refusalFrom, sessionHeaders} from "./pilot-client.js";

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
  const sourceField = form.querySelector(".source-text-field");
  const sourceText = document.getElementById("source-text");
  const dropzone = document.getElementById("upload-dropzone");
  const preview = document.getElementById("upload-preview");
  const previewBadge = document.getElementById("preview-badge");
  const previewRemove = document.getElementById("preview-remove");
  const fileName = document.getElementById("upload-file-name");
  const fileDetail = document.getElementById("upload-file-detail");
  const submitButton = document.getElementById("evaluate-assignment");
  const feedback = document.getElementById("submission-feedback");
  const workspace = document.getElementById("upload-workspace");
  const progressPhase = document.getElementById("evaluation-phase");
  const progressFill = document.querySelector(".progress-fill");
  const progressPercent = document.getElementById("evaluation-percent");
  const result = document.getElementById("assessment-result");
  const resultHeadline = document.getElementById("assessment-headline");
  const resultLink = document.getElementById("result-link");
  const reportButton = document.getElementById("download-report");
  const copyButton = document.getElementById("copy-report-link");
  const cancelButton = document.getElementById("cancel-submission");
  const dialogStatus = document.getElementById("dialog-status");

  // The pilot has one recognition model switched on — Russian (the Uzbek button carries
  // data-unavailable and is disabled) — and the enable rule below accepts nothing else. Naming it
  // once, here, is what lets the pressed button, state.language and the request read the same fact:
  // the pressed state used to be left to the markup of two pages while the rule read the state, and
  // on the Russian page the two disagreed until a teacher clicked «Русский» herself (2026-09-17).
  const PILOT_LANGUAGE = "ru";

  // And one assignment type switched on — dictation; the other four carry data-unavailable and are
  // disabled. Same rule, same reason: a group with exactly one selectable option opens with it
  // selected. Measured on the live site 2026-09-17: a teacher pasted her key, uploaded her page and
  // «Проверить работу» stayed grey, because the type had never been pressed — the one control she
  // had no reason to think was a choice.
  const PILOT_ASSIGNMENT = "dictation";

  const state = {
    language: PILOT_LANGUAGE,
    assignment: PILOT_ASSIGNMENT,
    file: null,
    previewUrl: "",
    busy: false,
    jobId: "",
    reportUrl: "",
    reportFilename: "",
    publicUrl: "",
    availableUntil: "",
    handedOutReportUrl: "",
    pageScrollY: 0,
    copyTimer: 0,
    cancelLockTimer: 0,
    pollGeneration: 0,
    // Where the bar is: which stage's band it is in, when it entered that band, the place in the line
    // the gateway last reported, and the highest fraction it has shown. The last one is what makes
    // the bar unable to go backwards, whatever arrives out of order (2026-09-17).
    waitStage: 0,
    waitStageSince: 0,
    queuePosition: 0,
    barFraction: 0,
    waitTimer: 0,
  };

  // The Russian page (docs/index.html) and the English one (docs/en/index.html) load this one
  // module, so the wording follows the page's own `lang`. Everything the form says outside these
  // live states is already written out in the markup of each page — including the first line the
  // waiting panel shows, which is read back off the page below.
  const english = document.documentElement.lang === "en";

  // Every live string this module writes onto the screen, in the language of the page that loaded
  // it. Nothing here names the machinery behind the pilot: a teacher waiting for her page gets a
  // sentence about her page, and the words that describe the inside of the system stay out of the
  // user-visible text (2026-09-17).
  //
  // The waiting panel has one line and one sub-line, and neither of them moves: the phase used to be
  // rebuilt from the gateway's status and from the job's place in the line, and a sentence that
  // changes under a waiting teacher reads as a glitch rather than as news (2026-09-17). The line is
  // the founder's, and the page's own half of it sits in the markup under this one.
  const ui = english
    ? {
        sending: "Sending your page…",
        waitingLine: "We read it three times too, at first.",
        waiting: "Your page is accepted.",
        done: "Done",
        nextPage: "Paste the teacher's text and upload the next page.",
        needBoth: "Paste the teacher's text and upload a photo.",
        needText: "The photo is uploaded. Paste the teacher's text.",
        needPhoto: "The teacher's text is in. Upload the photo of the page.",
        ready: "Everything is in place. Press “Check the work”.",
        notServed: "Only Russian dictations are checked for now.",
        noFile: "No file selected",
        fileHint: "One well-lit page with no cropped edges",
        wrongFormat: "Upload a photo in JPG, PNG or WEBP format.",
        tooBig: "The file is larger than 20 MB. Make the photo smaller and try again.",
        replaceHint: "click the page to replace it",
        uploadedWork: "Uploaded work",
        checkedWork: "Work checked",
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
        deleteFailed: "Could not delete the report. Please try again.",
        downloadFailed: "Could not download the report. Please try again.",
        copyManual: "Copy from the field below",
        resultFailed: "Could not get the check result.",
        resultHeadline: "Check complete",
        confirmDelete: "Delete the report?",
        availableTill: "Available till",
        copied: "Link copied",
      }
    : {
        sending: "Отправляем вашу страницу…",
        waitingLine: "Мы тоже сначала читаем по три раза.",
        waiting: "Работа принята.",
        done: "Готово",
        nextPage: "Вставьте текст учителя и загрузите следующую страницу.",
        needBoth: "Вставьте текст учителя и загрузите фотографию.",
        needText: "Фотография загружена. Осталось вставить текст учителя.",
        needPhoto: "Текст учителя вставлен. Осталось загрузить фотографию страницы.",
        ready: "Всё на месте. Нажмите «Проверить работу».",
        notServed: "Пока проверяются только русские диктанты.",
        noFile: "Файл не выбран",
        fileHint: "Одна хорошо освещённая страница без обрезанных краёв",
        wrongFormat: "Загрузите фотографию в формате JPG, PNG или WEBP.",
        tooBig: "Файл больше 20 МБ. Уменьшите фотографию и попробуйте снова.",
        replaceHint: "нажмите на страницу, чтобы заменить",
        uploadedWork: "Загруженная работа",
        checkedWork: "Работа проверена",
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
        deleteFailed: "Не удалось удалить отчёт. Попробуйте ещё раз.",
        downloadFailed: "Не удалось скачать отчёт. Попробуйте ещё раз.",
        copyManual: "Скопируйте из поля ниже",
        resultFailed: "Не удалось получить результат проверки.",
        resultHeadline: "Проверка завершена",
        confirmDelete: "Удалить отчёт?",
        availableTill: "Доступно до",
        copied: "Ссылка скопирована",
      };

  // The panel's first line is the page's own sentence for the seconds the photo is on its way up,
  // read off the markup so both locales carry it in one place. A second check in the same page
  // reuses it instead of keeping whatever the first one ended on.
  const sendingPhase = progressPhase.textContent;

  // The same trick for the two buttons whose resting label is the page's own word, read back off the
  // markup: the cancel button's two pressed states and the copy button's reveal and confirmation all
  // have exactly one resting state to return to, and no second copy of «Отмена» or «Скопировать
  // ссылку» lives in this module (2026-09-17).
  const cancelLabel = cancelButton.textContent;
  const copyLabel = copyButton.textContent;

  function endpoint() {
    const configured = document.querySelector('meta[name="bilimai-api-url"]')?.content.trim() || "";
    const localOverride = ["127.0.0.1", "localhost"].includes(window.location.hostname)
      ? new URLSearchParams(window.location.search).get("api") || ""
      : "";
    const value = configured || localOverride;
    return value.endsWith("/") ? value.slice(0, -1) : value;
  }

  const apiBase = endpoint();

  // 429 is the one refusal the teacher can act on, so it is the one that says how long to wait. Two
  // scopes reach here now, her own allowance and her address's; the waiting room's own refusal was
  // retired with the room's cap (founder, 2026-09-17), so the branch that spoke for it is gone. The
  // gateway's own sentence is not repeated: it names what ran out, and a teacher reading this form
  // is never told about that — the number is the gateway's, said as a wait.
  function refusalMessage(refusal) {
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

  /* The one condition that opens the form: the pilot's own pair of settings, the teacher's text and
     her photograph. Both the button's `disabled` and the line under it are read from it, so the two
     cannot come to disagree about whether there is anything left to ask her for. */
  function readyToSend() {
    return state.language === PILOT_LANGUAGE
      && state.assignment === PILOT_ASSIGNMENT
      && Boolean(sourceText.value.trim())
      && Boolean(state.file);
  }

  function updateSubmitState() {
    submitButton.disabled = state.busy || !readyToSend();
  }

  /* The line under the form answers one question — what is still missing — because the sentence it
     used to carry asked the teacher to «подтвердить параметры», and the settings it named are
     already made by the two preselections above: a teacher with her key pasted and her photograph
     uploaded saw a grey «Проверить работу» and a line telling her to do what she had done (founder,
     2026-09-17). It never lists a thing she has supplied, and it is written from the page's own
     locale, so the two index.html files carry no copy of it that could drift from this rule. */
  function announceReadiness() {
    const text = sourceText.value.trim();
    if (!text && !state.file) setFeedback(ui.needBoth);
    else if (!text) setFeedback(ui.needText);
    else if (!state.file) setFeedback(ui.needPhoto);
    // Both of hers are in, so the form is open unless the pressed pair is not one this pilot serves.
    // Only a markup change can produce that — both pages ship exactly one selectable option in each
    // group, so the press can only ever re-state the pilot's own pair — and the line has to be as
    // honest in that state as in the other three, or it says "everything is in place" over a grey
    // button.
    else setFeedback(readyToSend() ? ui.ready : ui.notServed);
  }

  function pressSingle(buttons, selected, attribute) {
    buttons.forEach((button) => {
      button.setAttribute("aria-pressed", String(button.getAttribute(attribute) === selected));
    });
  }

  /* The teacher's own material — her photograph and her pasted text — while a submission is in
     flight, and the X that would clear the photograph, which is part of the same material: the page
     it stands on is already on its way to a teacher, so clearing it mid-check would leave him
     reading a page the form no longer shows. `inert` is the whole mechanism: one attribute takes
     them out of the tab order, out of the accessibility tree and off the pointer, so "blurred" and
     "unreachable" cannot drift apart. The class on the workspace is only what the CSS blurs and
     hides (2026-09-17). */
  function parkMaterial(parked) {
    workspace.classList.toggle("is-parked", parked);
    [sourceField, dropzone, previewRemove].forEach((element) => element.toggleAttribute("inert", parked));
  }

  function setDialogStatus(message) {
    dialogStatus.textContent = message;
    dialogStatus.hidden = !message;
  }

  /* The badge over the photograph names one of two different pages: the one she uploaded, or the one
     BilimAI marked. Green is what "checked" looks like in this page's own palette, so the colour
     belongs to the checked badge alone, and the words and the colour are set together, in one place
     (2026-09-17). */
  function setPreviewBadge(checked) {
    previewBadge.textContent = checked ? ui.checkedWork : ui.uploadedWork;
    previewBadge.classList.toggle("is-checked", checked);
  }

  /* The page behind the dialog does not move while the dialog is up: not by wheel, not by trackpad,
     not by a touch drag, not by the arrow keys or the space bar, and not by a focus move trying to
     drag the page to something behind it. `overflow: hidden` on the root (in submit.css) is what
     stops the scroll; the rest of this is the two defects that trick is known for. A disappearing
     scrollbar drags every element on the page sideways by its width, so the width is measured before
     the lock and given back as padding. And iOS Safari scrolls the page behind a modal anyway, so the
     page is pinned at the offset it was at instead of being left to the viewport (2026-09-17). */
  function lockPageScroll() {
    const root = document.documentElement;
    state.pageScrollY = window.scrollY;
    const barWidth = window.innerWidth - root.clientWidth;
    root.classList.add("is-scroll-locked");
    if (barWidth > 0) document.body.style.paddingRight = `${barWidth}px`;
    document.body.style.position = "fixed";
    document.body.style.top = `${-state.pageScrollY}px`;
    document.body.style.width = "100%";
  }

  function unlockPageScroll() {
    const root = document.documentElement;
    root.classList.remove("is-scroll-locked");
    document.body.style.removeProperty("padding-right");
    document.body.style.removeProperty("position");
    document.body.style.removeProperty("top");
    document.body.style.removeProperty("width");
    // Put back on the pixel it was on, with the page's own smooth scrolling held out of it: the
    // teacher did not ask to travel back to where she was (2026-09-17).
    root.style.scrollBehavior = "auto";
    window.scrollTo(0, state.pageScrollY);
    root.style.removeProperty("scroll-behavior");
  }

  /* The end of the flow, whichever way it ended: the report is the browser's business now (or was
     deleted), and the teacher gets her form back — empty, unblurred and ready for the next page.
     The photograph leaves through releasePhoto() and not by hand, which is why nothing clears the
     result before a download has been handed over (2026-09-17). */
  function clearSubmission() {
    state.pollGeneration += 1;
    state.jobId = "";
    state.reportUrl = "";
    state.reportFilename = "";
    if (result.open) result.close();
    // Back to rest on both buttons, which also drops the lock's and the copy confirmation's pending
    // timers and puts the clipboard's fallback field away: none of them outlives the dialog they were
    // started from (2026-09-17).
    setCancelState("resting");
    setCopyState("resting");
    setDialogStatus("");
    sourceText.value = "";
    releasePhoto();
    parkMaterial(false);
    setFeedback(ui.nextPage);
    updateSubmitState();
    /* Focus goes to the first field of the empty form: a closed modal hands the keyboard back to
       the page behind it, and the next thing this form asks for is the teacher's text (2026-09-17). */
    sourceText.focus({preventScroll: true});
  }

  function formatSize(bytes) {
    if (bytes < 1024 * 1024) return `${Math.max(1, Math.round(bytes / 1024))} ${ui.kilobytes}`;
    return `${(bytes / (1024 * 1024)).toFixed(1)} ${ui.megabytes}`;
  }

  function isAcceptedImage(file) {
    return ["image/jpeg", "image/png", "image/webp"].includes(file.type);
  }

  /* Everything the form is holding about the photograph, dropped in one place: her file, the object
     URL the preview is drawn from, the picture, the badge over it, the two lines under it and the
     class that says a page is attached. One function because the photograph leaves in three ways —
     she picks another one, the form refuses the one she picked, and she presses the X on it — and a
     URL dropped without being revoked holds its whole file for as long as the tab lives, up to the
     form's own 20 MB (2026-09-17). */
  function releasePhoto() {
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
  }

  function rejectFile(message) {
    releasePhoto();
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

    releasePhoto();
    state.file = file;
    state.previewUrl = URL.createObjectURL(file);
    preview.src = state.previewUrl;
    preview.alt = `${ui.uploadedWork}: ${file.name}`;
    setPreviewBadge(false);
    dropzone.classList.add("has-file");
    fileName.textContent = file.name;
    fileDetail.textContent = `${formatSize(file.size)} · ${ui.replaceHint}`;
    announceReadiness();
    updateSubmitState();
  }

  /* The X above the photograph: her page leaves the form, and nothing else does — the teacher's text
     is still hers. The two things that speak for the file, the line under the form and the button
     beside it, are brought up to date by the same two functions every other change to this form goes
     through, so there is no second copy of the rule that opens the button (2026-09-17). */
  function removePhoto() {
    releasePhoto();
    announceReadiness();
    updateSubmitState();
    // The X goes with the photograph it stood on, so the keyboard lands on the control that puts one
    // back instead of at the top of the page (2026-09-17).
    dropzone.focus({preventScroll: true});
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

  /* The waiting bar, one band per stage the gateway names. One time curve cannot serve both pages
     this form has to be honest about: it was tuned for the cold start the pilot measured at 10-15
     minutes (2026-09-17), where the founder watched it crawl, and it described the estimate rather
     than the page — a warm page is ~3.5 s of GPU compute inside ~11 s end to end, which that curve
     read as 2-3 % before jumping to done. So each stage owns a band of the bar, entered the moment
     the gateway reports it and filled by time inside that band, and the bands are sized for the range
     between those two pages:
   
     * `queued` takes 40 % of the bar on a 7-minute time constant, which is where the cold start's
       minutes live: it is the one stage whose length is not a property of the page at all, so it is
       the one band that has to creep for minutes without parking at a number. A page that is 3rd in
       line reads half of what a page 1st reads, because it is genuinely further from done.
     * `starting` (40 → 55 %) and `rendering` (90 → 99 %) are the two short ends of the check, and
       `evaluating` (55 → 90 %) takes the largest of the three because the compute is the largest
       measured piece of a warm page — so a warm page is past halfway the moment it is evaluating,
       which is the number the founder did not have.
   
     Each band then moves on a curve whose time constant is that stage's own measured duration, so a
     warm page leaves a stage with about two-thirds of its band filled, and a stage that runs long
     keeps creeping for several more seconds instead of parking the moment the measurement is spent.
   
     Nothing moves the bar backwards: every band opens where the one above it closes, and a stage the
     bar has already left is not news. Nothing prints 100 but the gateway saying the check is
     complete: the last band stops at 0.99, whose honest whole-number rounding is 99. */
  const WAIT_TICK_MS = 250;                  // sub-pixel steps on a 440px card, for a quarter of the work of rAF
  const WAIT_BANDS = [
    {status: "queued", ceiling: 0.40, tauMs: 7 * 60 * 1000, queuePositioned: true},
    {status: "starting", ceiling: 0.55, tauMs: 1500},
    {status: "evaluating", ceiling: 0.90, tauMs: 3500},
    {status: "rendering", ceiling: 0.99, tauMs: 1500},
  ];

  // Each band opens where the one above it closes: a boundary is one number, stated once.
  WAIT_BANDS.forEach((band, index) => { band.floor = WAIT_BANDS[index - 1]?.ceiling ?? 0; });

  // Built once, not four times a second, and from the page's own locale, which is the only thing
  // that decides whether the number reads "13 %" or "13%".
  const percentFormat = new Intl.NumberFormat(ui.locale, {style: "percent", maximumFractionDigits: 0});

  // The fill and the number are written from one fraction, on one repaint, so the readout can never
  // describe a bar other than the one on screen. No band's own fraction reaches its last ceiling of
  // 0.99, whose honest whole-number rounding is 99, so nothing but finishWaiting()'s 1 can print 100
  // — the number says the same thing about the wait that the bar does (2026-09-17).
  function paintBar(fraction) {
    const percent = Math.round(fraction * 100);
    progressFill.style.transform = `scaleX(${fraction.toFixed(4)})`;
    progressPercent.textContent = percentFormat.format(percent / 100);
    // The track is aria-hidden, so the number is what a screen reader has: it carries the value as
    // role="progressbar", named by the phase line beside it rather than by copy of its own.
    progressPercent.setAttribute("aria-valuenow", String(percent));
  }

  /* What one band has earned after `elapsedMs` in it: its share of the bar, filled on a curve with
     the stage's own measured duration as its time constant. A page holding a place in the line is not
     as far along as one that is first — the queue's band is scaled by that place, 1st in line reading
     a whole band and 3rd reading half of it — and both keep creeping, because neither the gateway nor
     this page knows when the machine arrives (2026-09-17). */
  function bandFraction(band, elapsedMs, queuePosition) {
    const place = band.queuePositioned && queuePosition > 0 ? 2 / (queuePosition + 1) : 1;
    const earned = 1 - Math.exp(-elapsedMs / band.tauMs);
    return band.floor + (band.ceiling - band.floor) * place * earned;
  }

  function paintWait() {
    const band = WAIT_BANDS[state.waitStage];
    const earned = bandFraction(band, Date.now() - state.waitStageSince, state.queuePosition);
    // The one written-down fraction, and the whole of the "never backwards" rule: a later poll, an
    // earlier stage, a place in the line that got worse — whichever arrives, the bar only ever takes
    // the larger value (2026-09-17).
    state.barFraction = Math.max(state.barFraction, earned);
    paintBar(state.barFraction);
  }

  /* The gateway's word about where the page is: the stage it names picks the band and starts that
     band's clock, and the place in the line it names is what the queue's band is scaled by. A status
     this bar has no band for says nothing, and a stage the bar has already left is not news and does
     not move it — a poll answered out of order cannot take the number back down (2026-09-17). */
  function reportStage(status, queuePosition) {
    const index = WAIT_BANDS.findIndex((band) => band.status === status);
    if (index < 0) return;
    if (queuePosition > 0) state.queuePosition = queuePosition;
    if (index > state.waitStage) {
      state.waitStage = index;
      state.waitStageSince = Date.now();
    }
    paintWait();   // the band's entry value, on the poll that brought the news
  }

  // Called once the gateway has accepted the page: until then the panel names no progress it cannot
  // know about, which is why a refusal never gets to show a moving bar.
  function startWaiting() {
    state.waitStage = 0;
    state.waitStageSince = Date.now();
    state.queuePosition = 0;
    state.barFraction = 0;
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
    paintBar(1);
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

  /* Two seconds is this dialog's beat: the cooling-off the cancel button serves after its first
     press, and how long a copy confirmation stays up before the button is a button again
     (2026-09-17). */
  const DIALOG_BEAT_MS = 2000;

  /* The link's two presses are the cancel button's idea without the danger: the first one gives the
     teacher the fact she needs before she sends the link to anyone — how long it stays alive, taken
     from the result's own expiry rather than from a sentence about it — and the second one copies it.
     A press that has not been told the deadline yet copies nothing (2026-09-17). */
  function setCopyState(value) {
    window.clearTimeout(state.copyTimer);
    state.copyTimer = 0;
    if (value === "resting") {
      // Back to rest puts the fallback field away too, in the same place that opened it (2026-09-17).
      copyButton.classList.remove("is-revealed");
      resultLink.hidden = true;
      resultLink.value = "";
    } else {
      copyButton.classList.add("is-revealed");
    }
    copyButton.textContent = {
      resting: copyLabel,
      revealed: `${ui.availableTill} ${state.availableUntil}`,
      copied: ui.copied,
      manual: ui.copyManual,
    }[value];
    // The confirmation is a beat, not a state: it says what just happened and then the button is the
    // button again. Two seconds is this dialog's beat, the same one the cancel lock uses (2026-09-17).
    if (value === "copied") {
      state.copyTimer = window.setTimeout(() => setCopyState("resting"), DIALOG_BEAT_MS);
    }
  }

  async function copyReportLink() {
    if (!copyButton.classList.contains("is-revealed")) {
      setCopyState("revealed");
      return;
    }
    try {
      // Over HTTPS this is the whole job. It is refused on a page that is not a secure context and by
      // a browser that will not grant the permission, and both of those are answered below rather
      // than with an error: the teacher came here for this URL (2026-09-17).
      await navigator.clipboard.writeText(state.publicUrl);
      setCopyState("copied");
    } catch {
      // The link itself, on screen and selected, is the fallback: it can be copied by hand from here.
      resultLink.value = state.publicUrl;
      resultLink.hidden = false;
      resultLink.focus({preventScroll: true});
      resultLink.select();
      setCopyState("manual");
    }
  }

  /* The dialog is the only place the report can be reached from, so it opens holding everything a
     press needs: the three URLs the buttons hand over, the deadline that is the copy button's second
     label, and the buttons themselves. It is one dialog that is reopened rather than rebuilt, so
     every opening starts from the same state — every button live and at rest, no error and no link
     left over from a failed press (2026-09-17). */
  function showResult(payload) {
    const evaluatedUrl = safeUrl(payload.evaluated_document_url, "image");
    const reportUrl = safeUrl(payload.report_pdf_url, "pdf");
    const publicUrl = safeUrl(payload.public_url, "page");
    const expires = new Date(payload.expires_at);
    const assessment = payload.assessment;
    // The expiry is one of the things the dialog needs, not decoration: it is the copy button's own
    // second label, and the gateway sends one with every completed job (2026-09-17).
    if (
      !evaluatedUrl
      || !reportUrl
      || !publicUrl
      || Number.isNaN(expires.getTime())
      || !assessment
      || typeof assessment !== "object"
    ) {
      throw new Error(ui.incompleteResult);
    }

    preview.src = evaluatedUrl;
    preview.alt = ui.checkedAlt;
    setPreviewBadge(true);
    resultHeadline.textContent = String(assessment.headline || ui.resultHeadline);

    state.reportUrl = reportUrl;
    state.reportFilename = String(payload.report_filename || "bilimai-report.pdf");
    state.publicUrl = publicUrl;
    // Formatted here, once: the copy button's reveal and nothing else on the page says when the
    // result stops existing (2026-09-17).
    state.availableUntil = expires.toLocaleTimeString(ui.locale, {hour: "2-digit", minute: "2-digit"});

    reportButton.disabled = false;
    copyButton.disabled = false;
    cancelButton.disabled = false;
    setCancelState("resting");
    setCopyState("resting");
    setDialogStatus("");
    lockPageScroll();
    result.showModal();
    /* The teacher came here for the report, so the dialog opens with the primary action focused:
       Enter downloads it, and Tab reaches the copy and the cancel beside it (2026-09-17). */
    reportButton.focus({preventScroll: true});
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
        return;
      }
      if (payload.status === "failed" || payload.status === "cancelled") {
        throw new Error(payload.error || ui.checkFailed);
      }
      // What the gateway reports while a job runs is not put on screen as words: the panel says one
      // thing for the whole wait, and the bar is what carries the news. Its status and its place in
      // the line are the only two facts there are about where the page actually is, and they are
      // what the bar is drawn from (2026-09-17).
      reportStage(payload.status, payload.queue_position);
      await new Promise((resolve) => window.setTimeout(resolve, 1500));
    }
  }

  languageButtons.forEach((button) => {
    if (button.dataset.unavailable) return;
    button.addEventListener("click", () => {
      state.language = button.dataset.language;
      pressSingle(languageButtons, state.language, "data-language");
      announceReadiness();
      updateSubmitState();
    });
  });

  categoryButtons.forEach((button) => {
    if (button.dataset.unavailable) return;
    button.addEventListener("click", () => {
      state.assignment = button.dataset.assignment;
      pressSingle(categoryButtons, state.assignment, "data-assignment");
      announceReadiness();
      updateSubmitState();
    });
  });

  // Her text is the one input whose readiness nothing else re-reads, so the line follows the typing
  // rather than the last event that happened to pass through: paste the key first and the line names
  // the photograph alone (2026-09-17).
  sourceText.addEventListener("input", () => {
    announceReadiness();
    updateSubmitState();
  });
  fileInput.addEventListener("change", () => loadFile(fileInput.files?.[0]));
  previewRemove.addEventListener("click", removePhoto);
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

    progressPhase.textContent = sendingPhase;
    setFeedback(ui.sending);
    setBusy(true);
    // Parked before the first byte leaves: the moment the teacher presses the button, her photograph
    // and her text stop being hers to edit and start being what is being checked (2026-09-17).
    parkMaterial(true);
    const body = new FormData();
    body.append("document", state.file, state.file.name);
    body.append("source_text", sourceText.value.trim());
    body.append("language", PILOT_LANGUAGE);
    body.append("assignment_type", PILOT_ASSIGNMENT);

    try {
      const accepted = await fetchJson("/v1/submissions", {method: "POST", body});
      state.jobId = String(accepted.job_id || "");
      if (!state.jobId) throw new Error(ui.noJobId);
      const generation = ++state.pollGeneration;
      startWaiting();
      // One line for the whole wait, and the page's own sub-line under it. The live region carries
      // the part a screen reader cannot see change: that the page was accepted.
      progressPhase.textContent = ui.waitingLine;
      setFeedback(ui.waiting);
      await pollJob(state.jobId, generation);
    } catch (error) {
      // Nothing is running any more, so her material goes back to being hers: a check that was not
      // accepted, or that failed, has to leave a form she can correct and send again — not a blurred
      // photograph (2026-09-17).
      parkMaterial(false);
      setFeedback(error instanceof Error ? error.message : ui.resultFailed, true);
    } finally {
      stopWaiting();
      setBusy(false);
    }
  });

  /* Cancel throws the report away for good, so it has three states rather than a plain confirm: the
     yellow it rests in, a two-second lock that says what the next press would do and cannot itself be
     pressed, and the red that does it. The lock is a real `disabled` button — one that looks pressable
     and does nothing reads as broken — and its own timer is the only thing that ends it, so a press
     that lands a moment too late cannot fall through to the delete. Escape puts either state back to
     rest; nothing else does, because the lock already covers the moment the first press bought
     (2026-09-17). */
  function setCancelState(value) {
    window.clearTimeout(state.cancelLockTimer);
    state.cancelLockTimer = 0;
    const locked = value === "locked";
    cancelButton.classList.toggle("is-locked", locked);
    cancelButton.classList.toggle("is-armed", value === "armed");
    cancelButton.textContent = value === "resting" ? cancelLabel : ui.confirmDelete;
    cancelButton.disabled = locked;
    // `disabled` is what makes the press impossible; this says the same thing in the markup, so the
    // locked button is described as unavailable rather than as a button that happens not to work.
    if (locked) cancelButton.setAttribute("aria-disabled", "true");
    else cancelButton.removeAttribute("aria-disabled");
    if (!locked) return;
    state.cancelLockTimer = window.setTimeout(() => {
      setCancelState("armed");
      // The lock takes the button out of the tab order, which drops focus out of it for those two
      // seconds: it is given back when the button is clickable again, unless the teacher has moved on
      // to something else inside the dialog in the meantime (2026-09-17).
      if (!result.contains(document.activeElement)) cancelButton.focus({preventScroll: true});
    }, DIALOG_BEAT_MS);
  }

  async function cancelSubmission() {
    // A press that lands during the lock is not a press: the button is already disabled, and an
    // impatient second click must not restart the two seconds, let alone reach the delete
    // (2026-09-17).
    if (cancelButton.disabled) return;
    if (!cancelButton.classList.contains("is-armed")) {
      setCancelState("locked");
      return;
    }
    setCancelState("resting");
    reportButton.disabled = true;
    cancelButton.disabled = true;
    try {
      // A completed job is the only way into this dialog, so there is always a job to discard.
      await fetchJson(`/v1/submissions/${encodeURIComponent(state.jobId)}`, {method: "DELETE"});
      clearSubmission();
    } catch (error) {
      // The report is still there — the gateway never confirmed the delete — so the dialog stays
      // open with the reason on it, and the teacher can press again (2026-09-17). Focus goes back to
      // the button that failed: disabling it while it held focus would leave the keyboard nowhere.
      setDialogStatus(error instanceof Error ? error.message : ui.deleteFailed);
      reportButton.disabled = false;
      cancelButton.disabled = false;
      cancelButton.focus({preventScroll: true});
    }
  }

  /* The report reaches the teacher before the form forgets anything: the bytes are fetched in full,
     handed to the browser as a blob URL with the gateway's own filename, and only then does the flow
     reset. Clearing first — revoking the photograph's URL and dropping the result — would send her
     after a file that no longer exists. The click hands the blob to the download manager, and the
     blob URL outlives it: the next report releases the previous URL rather than the reset releasing
     the one being read (2026-09-17). */
  async function downloadReport() {
    const reportUrl = state.reportUrl;
    if (!reportUrl || reportButton.disabled) return;
    reportButton.disabled = true;
    cancelButton.disabled = true;
    try {
      const response = await fetch(reportUrl);
      if (!response.ok) throw new Error(ui.downloadFailed);
      const url = URL.createObjectURL(await response.blob());
      const anchor = document.createElement("a");
      anchor.href = url;
      anchor.download = state.reportFilename;
      anchor.click();
      if (state.handedOutReportUrl) URL.revokeObjectURL(state.handedOutReportUrl);
      state.handedOutReportUrl = url;
      clearSubmission();
    } catch (error) {
      setDialogStatus(error instanceof Error ? error.message : ui.downloadFailed);
      reportButton.disabled = false;
      cancelButton.disabled = false;
      reportButton.focus({preventScroll: true});
    }
  }

  reportButton.addEventListener("click", downloadReport);
  copyButton.addEventListener("click", copyReportLink);
  cancelButton.addEventListener("click", cancelSubmission);
  // The lock belongs to the dialog's visibility, not to any one way out of it: whatever closes the
  // dialog gives the page its scrolling back, so a request that fails cannot leave a page that
  // refuses to move (2026-09-17).
  result.addEventListener("close", unlockPageScroll);
  result.addEventListener("cancel", (event) => {
    // Escape never closes this dialog: closing it would throw the report away without a word, and the
    // report only ever leaves through the buttons. It does put both pressed buttons back to rest —
    // the copy button's reveal, and the cancel button's lock or its armed state (2026-09-17).
    event.preventDefault();
    setCopyState("resting");
    setCancelState("resting");
  });

  // The initial selection, mirrored onto the buttons from the one place each is stated above: both
  // pages open with «Русский» and «Диктант» pressed, so the pressed state and the enable rule agree
  // from first paint and the form is submittable without either click (2026-09-17).
  pressSingle(languageButtons, PILOT_LANGUAGE, "data-language");
  pressSingle(categoryButtons, PILOT_ASSIGNMENT, "data-assignment");
  announceReadiness();
  updateSubmitState();
})();
