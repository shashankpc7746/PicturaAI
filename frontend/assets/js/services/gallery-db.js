// "My gallery": saved paintings in IndexedDB (this device only).
// Record: { id, createdAt, title, subtitle, intensity, mode, width, height,
//           image: Blob (final JPEG), thumb: Blob, photo?: Blob, stylized?: Blob }

const DB = 'pictura-gallery';
const STORE = 'paintings';
let dbPromise = null;

function open() {
  if (!dbPromise) {
    dbPromise = new Promise((resolve, reject) => {
      if (!('indexedDB' in window)) { reject(new Error('Saving needs a browser with IndexedDB.')); return; }
      const req = indexedDB.open(DB, 1);
      req.onupgradeneeded = () => {
        const os = req.result.createObjectStore(STORE, { keyPath: 'id' });
        os.createIndex('createdAt', 'createdAt');
      };
      req.onsuccess = () => resolve(req.result);
      req.onerror = () => reject(req.error || new Error('Could not open the gallery.'));
    });
    dbPromise.catch(() => { dbPromise = null; });
  }
  return dbPromise;
}

async function tx(mode, fn) {
  const db = await open();
  return new Promise((resolve, reject) => {
    const t = db.transaction(STORE, mode);
    const os = t.objectStore(STORE);
    let out;
    Promise.resolve(fn(os)).then((v) => { out = v; });
    t.oncomplete = () => resolve(out);
    t.onerror = () => reject(t.error || new Error('Gallery storage failed.'));
    t.onabort = () => reject(t.error || new Error('Gallery storage was interrupted. Your device may be out of space.'));
  });
}

const req2p = (req) => new Promise((resolve, reject) => {
  req.onsuccess = () => resolve(req.result);
  req.onerror = () => reject(req.error);
});

export const putPainting = (rec) => tx('readwrite', (os) => { os.put(rec); return rec.id; });
export const deletePainting = (id) => tx('readwrite', (os) => { os.delete(id); });
export const clearPaintings = () => tx('readwrite', (os) => { os.clear(); });

export async function listPaintings() {
  const db = await open();
  const all = await req2p(db.transaction(STORE).objectStore(STORE).getAll());
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
