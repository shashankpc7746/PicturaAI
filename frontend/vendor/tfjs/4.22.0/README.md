# TensorFlow.js 4.22.0 (self-hosted)

Unmodified copies of the ES2017 UMD builds from npm, used by the on-device engine
(`frontend/assets/js/engine/browser-engine.js`). They are served from our own origin
because the engine runs in a module Web Worker, where `importScripts()` is unavailable
and a dynamic `import()` cannot carry Subresource Integrity. Self-hosting keeps the
exact bytes under version control instead.

| File | Source | SHA-384 (SRI) |
|---|---|---|
| `tf-core.es2017.min.js` | `@tensorflow/tfjs-core@4.22.0/dist/` | `sha384-pwFlZMvh0feP+i0lcqJy+EGMsa8y8MInwWkUgARHZEHUGyTNqxnqfi38fzMU994Z` |
| `tf-backend-webgl.es2017.min.js` | `@tensorflow/tfjs-backend-webgl@4.22.0/dist/` | `sha384-vLZ29FEcQ9QWLdF6cKts4jNSPRioiSHmaggrIjdw0FbgdUp3CV0qUb4Mvp1C/ehV` |
| `tf-converter.es2017.min.js` | `@tensorflow/tfjs-converter@4.22.0/dist/` | `sha384-jYgcd4ez9iEjyoeMMxPS7PCpt0FCX9Mc3ovkX5gEI0BnKmxJOiBBi9wWaPyzE40X` |

The hashes match the files on cdn.jsdelivr.net, and the main-thread fallback still
loads these files with `integrity` set to them. Use the `es2017` builds only:
`dist/tf.min.js` calls `Function()` at load time, which the site's CSP blocks.

The server caches this folder for a year (`immutable`), so an upgrade goes in a new
version folder (`frontend/vendor/tfjs/<version>/`); update `TF_VERSION` and the hashes in
`browser-engine.js` at the same time.

Licence: Apache-2.0, see `LICENSE` (from the tfjs repository at tag `tfjs-v4.22.0`).
