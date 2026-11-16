// Typed engine errors. `code` tells the UI what to do:
//   ENGINE_RESOURCE     the device ran out of GPU memory / lost the WebGL context:
//                       switch to the server engine (createEngine({ prefer: 'server' })).
//   ENGINE_UNAVAILABLE  the on-device engine could not load (blocked script, no WebGL2,
//                       missing model files): switch to the server engine.
//   ENGINE_FAILED       anything else inside the on-device engine: offer "Try again",
//                       or the server engine.
//   ENGINE_INPUT        bad input (no photo, unknown style): show the message.
//   SERVER_BUSY         429: ask to wait a moment.
//   SERVER_UNAVAILABLE  network error, 5xx, waking up: offer "Try again".
//   SERVER_REJECTED     other 4xx (e.g. 413 too large): show the message.
// AbortError (a DOMException) is thrown unchanged when the caller aborts.
export class EngineError extends Error {
  constructor(message, code, cause) {
    super(message);
    this.name = 'EngineError';
    this.code = code;
    if (cause) this.cause = cause;
  }
}
