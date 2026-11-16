// "My gallery": saved paintings in IndexedDB (this device only).
// Record: { id, createdAt, title, subtitle, intensity, mode, width, height,
//           image: Blob (final JPEG), thumb: Blob, photo?: Blob, stylized?: Blob }
//
// Errors that reach the page are always plain sentences: browsers report storage
// problems as DOMExceptions ("A mutation operation was attempted…") or TypeErrors.

const DB = 'pictura-gallery';
const STORE = 'paintings';
let dbPromise = null;

const UNAVAILABLE = "Saving isn't available in this browser, for example in private browsing. Download your painting instead.";
const FULL = 'Your browser is out of space for saved paintings. Delete some and try again.';
const FAILED = 'The gallery storage failed. Please try again.';

function galleryError(message, code, cause) {
  const err = new Error(message);
  err.code = code;
  if (cause) err.cause = cause;
  return err;
}

/** A friendly Error for any storage failure. code 'GALLERY_UNAVAILABLE' means saving cannot work here. */
function friendly(e) {
  if (e && e.code && String(e.code).startsWith('GALLERY_')) return e;
  const name = e && e.name;
  if (name === 'QuotaExceededError' || (e && /quota/i.test(e.message || ''))) return galleryError(FULL, 'GALLERY_FULL', e);
  if (name === 'SecurityError' || name === 'InvalidStateError') return galleryError(UNAVAILABLE, 'GALLERY_UNAVAILABLE', e);
  return galleryError(FAILED, 'GALLERY_FAILED', e);
}

function open() {
  if (!dbPromise) {
    dbPromise = new Promise((resolve, reject) => {
      // The property can exist and still be null (storage disabled in Firefox, some webviews).
      if (typeof window === 'undefined' || !window.indexedDB) {
        reject(galleryError(UNAVAILABLE, 'GALLERY_UNAVAILABLE'));
        return;
      }
      let req;
      try {
        req = window.indexedDB.open(DB, 1);
      } catch (e) {
        reject(galleryError(UNAVAILABLE, 'GALLERY_UNAVAILABLE', e));
        return;
      }
      req.onupgradeneeded = () => {
        const os = req.result.createObjectStore(STORE, { keyPath: 'id' });
        os.createIndex('createdAt', 'createdAt');
      };
      req.onsuccess = () => resolve(req.result);
      // Old Firefox private mode and blocked storage fail here.
      req.onerror = () => reject(galleryError(UNAVAILABLE, 'GALLERY_UNAVAILABLE', req.error));
    });
    dbPromise.catch(() => { dbPromise = null; });
  }
  return dbPromise;
}

async function tx(mode, fn) {
  const db = await open();
  return new Promise((resolve, reject) => {
    let t;
    try {
      t = db.transaction(STORE, mode);
    } catch (e) {
      reject(friendly(e));
      return;
    }
    const os = t.objectStore(STORE);
    let out;
    Promise.resolve(fn(os)).then((v) => { out = v; });
    t.oncomplete = () => resolve(out);
    t.onerror = () => reject(friendly(t.error));
    // An aborted write is most often a full disk.
    t.onabort = () => reject(t.error ? friendly(t.error) : galleryError(FULL, 'GALLERY_FULL'));
  });
}

const req2p = (req) => new Promise((resolve, reject) => {
  req.onsuccess = () => resolve(req.result);
  req.onerror = () => reject(friendly(req.error));
});

export const putPainting = (rec) => tx('readwrite', (os) => { os.put(rec); return rec.id; });
export const deletePainting = (id) => tx('readwrite', (os) => { os.delete(id); });
export const clearPaintings = () => tx('readwrite', (os) => { os.clear(); });

export async function listPaintings() {
  const db = await open();
  let all;
  try {
    all = await req2p(db.transaction(STORE).objectStore(STORE).getAll());
  } catch (e) {
    throw friendly(e);
  }
  return all.sort((a, b) => b.createdAt - a.createdAt);
}

export async function storageNote() {
  try {
    const { usage, quota } = await navigator.storage.estimate();
    if (!quota) return '';
    return `${(usage / 1048576).toFixed(1)} MB used of the space your browser allows.`;
  } catch {
    return '';
  }
}
