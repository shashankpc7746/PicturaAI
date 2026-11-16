// Minimal state container: get(), set(patch), subscribe(fn(state, prev)).

export function createStore(initial) {
  let state = { ...initial };
  const subs = new Set();
  return {
    get: () => state,
    set(patch) {
      const prev = state;
      const next = typeof patch === 'function' ? patch(state) : patch;
      state = { ...state, ...next };
      subs.forEach((fn) => fn(state, prev));
    },
    subscribe(fn) {
      subs.add(fn);
      return () => subs.delete(fn);
    },
  };
}

export const store = createStore({
  photo: null,          // { id, name, bytes, width, height, canvas, url, sample, caption, credit }
  styles: [],           // normalised presets (services/styles-data.js)
  stylesStatus: 'loading', // loading | waking | ready | error
  style: null,          // { source: 'preset'|'custom'|'generated'|'prompt', key?, title, artist?, year?, canvas?, prompt? }
  style2: null,         // preset for "Mix with a second style"
  mixOpen: false,
  mix: 0.3,             // share of the second style
  mode: 'full',         // full | palette | region
  intensity: 0.75,
  phase: 'idle',        // idle | generating | error | result
  error: null,          // { message, retry: bool, smaller: bool }
  result: null,         // { sig, kind: 'nst'|'palette', style, width, height }
  maskHasContent: false,
  engineKind: null,     // 'browser' | 'server' once known
});
