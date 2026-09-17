/**
 * What the two locales' pages must agree on about the public pilot, and nothing else.
 *
 * Both docs/index.html and docs/en/index.html load one submission client, docs/submit.js, which
 * picks its wording from the page's own lang. What that client must not decide for itself is the
 * protocol: which header carries the session identity, where the gateway hands a new one back, how
 * a 429 is read, and where a waiting job's place in the line lives. That is this module — no
 * wording, no rendering, so a page cannot drift from the contract in
 * local://public-pilot-contracts.md §HTTP surface on its own.
 *
 * The identity is not a secret: it only spends the allowance of whoever holds it, so it travels in
 * localStorage rather than a cookie (a cookie set by bilimai.waib.net is a third-party cookie that
 * Safari drops on neutrolink.github.io) and is sent as the X-Bilimai-Session request header.
 */

const SESSION_HEADER = "X-Bilimai-Session";

// One key for both locales: switching language must not hand the visitor a second hourly allowance,
// which is the whole point of an identity that outlives a page.
const SESSION_STORAGE_KEY = "bilimai-session";

// Safari in private mode blocks localStorage outright, and reading it can throw. The token then
// lives in this page's memory only: that costs a visitor her identity across reloads, never a job,
// and the gateway's per-address ceiling still applies to whoever submits.
let session = readStored();

function readStored() {
  try {
    return window.localStorage.getItem(SESSION_STORAGE_KEY) || "";
  } catch {
    return "";
  }
}

/** `extraHeaders` plus the session header, when this browser has an identity to send yet. */
export function sessionHeaders(extraHeaders = {}) {
  const headers = {...extraHeaders};
  if (session) headers[SESSION_HEADER] = session;
  return headers;
}

/**
 * Keep the identity a response carries: the header the gateway mints one in, or the `session_token`
 * of an accepted submission. Called for every response, refusals included — a visitor refused on
 * her very first attempt is still given an identity, and dropping it would reset her browser to a
 * fresh one on the next try.
 */
export function adoptSession(response, payload) {
  const issued = response.headers.get(SESSION_HEADER) || payload?.session_token || "";
  if (typeof issued !== "string" || !issued || issued === session) return;
  session = issued;
  try {
    window.localStorage.setItem(SESSION_STORAGE_KEY, issued);
  } catch {
    // Storage blocked: the token lives for this page load only.
  }
}

/**
 * A 429 as data — `{scope, retryAfter}` — or null for any other response.
 *
 * `retryAfter` is 0 when neither the body nor the header said how long to wait; the caller must not
 * fabricate a number from that, because "come back in NaN minutes" is worse than admitting the
 * gateway did not say.
 */
export function refusalFrom(response, payload) {
  if (response.status !== 429) return null;
  const seconds = Number(payload?.retry_after ?? response.headers.get("Retry-After"));
  return {
    scope: String(payload?.scope || ""),
    retryAfter: Number.isFinite(seconds) && seconds > 0 ? seconds : 0,
  };
}
