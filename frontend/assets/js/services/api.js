// HTTP to the PicturaAI backend, with messages a person can act on.
// Render's proxy answers 502/503 with an HTML page while the free instance wakes
// up or restarts, so a failed body is not always JSON.

const STATUS_MESSAGES = {
  413: 'That image is too large for the server. Please use a smaller photo.',
  429: 'The server is busy with other pictures. Please wait a moment and try again.',
  502: 'The server is waking up or restarting. Please try again in about a minute.',
  503: 'The server is waking up or restarting. Please try again in about a minute.',
  504: 'The server took too long to answer. Please try again in a moment.',
};

export class ApiError extends Error {
  constructor(message, status = 0) {
    super(message);
    this.name = 'ApiError';
    this.status = status;
    this.code = status === 429 ? 'SERVER_BUSY'
      : status === 0 || status >= 500 ? 'SERVER_UNAVAILABLE' : 'SERVER_REJECTED';
  }
}

export async function errorFromResponse(res) {
  let detail = '';
  try {
    const body = await res.json();
    detail = body.detail || body.error || '';
  } catch { /* HTML or empty body */ }
  if (typeof detail !== 'string') detail = '';   // FastAPI validation errors are arrays
  const known = STATUS_MESSAGES[res.status];
  const useKnown = res.status === 429 || res.status === 413 || res.status >= 502;
  return new ApiError((useKnown ? known : detail || known)
    || `Something went wrong on the server (${res.status}). Please try again.`, res.status);
}

export async function apiFetch(url, options = {}) {
  let res;
  try {
    res = await fetch(url, options);
  } catch (e) {
    if (e && e.name === 'AbortError') throw e;
    throw new ApiError(navigator.onLine === false
      ? 'You appear to be offline. Check your connection and try again.'
      : 'Could not reach the server. It may be waking up; please try again in a minute.', 0);
  }
  if (!res.ok) throw await errorFromResponse(res);
  try {
    return await res.json();
  } catch {
    throw new ApiError('The server sent an unreadable answer. Please try again.', 502);
  }
}

export const getStyles = () => apiFetch('/api/styles');

export async function describeStyle(prompt, signal) {
  const form = new FormData();
  form.append('prompt', prompt);
  return apiFetch('/api/describe-style', { method: 'POST', body: form, signal });
}

export function base64ToBlob(b64, type = 'image/jpeg') {
  const bin = atob(b64);
  const bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  return new Blob([bytes], { type });
}
