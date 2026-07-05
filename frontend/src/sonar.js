/**
 * REBOUND Sonar Engine — Web Audio API
 *
 * iOS-safe design:
 *   1. createAndUnlock() MUST be called synchronously inside a user gesture
 *      (tap handler), BEFORE any await. It creates the AudioContext, resumes
 *      it, and plays a 1-sample silent buffer — the classic iOS unlock.
 *   2. init() can then run its awaits (getUserMedia, fetch) freely.
 *   3. The mic stays permanently connected to a rolling buffer, so we never
 *      need to re-acquire the stream per scan.
 *
 * Capture window is latency-aware: Android WebAudio output latency can be
 * 100–300 ms, so the chirp may leave the speaker long after start(0).
 * We wait (outputLatency + window + margin) and grab a wide window, letting
 * server-side deconvolution find the direct path inside it.
 *
 * Payload is Float32 (half the bytes of Float64). Server accepts
 * audio_dtype: "float32" | "float64".
 */

var DEFAULT_CAPTURE_S = 0.45;   // wide window: chirp (20 ms) + echoes + jitter
var SCHEDULE_AHEAD_S = 0.03;    // schedule chirp slightly in the future
var EXTRA_MARGIN_MS = 60;       // safety margin after the window

/** Human-readable environment report — shown on the error screen so we can
 *  diagnose judges' phones without a connected debugger. */
export function envDiagnostics() {
  var AC = window["AudioContext"] || window["webkitAudioContext"];
  var parts = [
    "secureContext=" + (window.isSecureContext ? "yes" : "NO"),
    "mediaDevices=" + (navigator.mediaDevices && navigator.mediaDevices.getUserMedia ? "yes" : "NO"),
    "AudioContext=" + (AC ? "yes" : "NO"),
    "speech=" + ("speechSynthesis" in window ? "yes" : "no"),
    "vibrate=" + (navigator.vibrate ? "yes" : "no"),
  ];
  return parts.join(" · ");
}

export function SonarEngine() {
  this.audioCtx = null;
  this.chirpBuffer = null;
  this.chirpSampleRate = 44100;
  this.stream = null;
  this.ready = false;
  this.captureS = DEFAULT_CAPTURE_S;
  this.extraLatencyMs = 0; // manual calibration via ?lat=NN (ms)

  // Rolling buffer — always recording once init() completes
  this._rollingBuffer = null;
  this._rollingPos = 0;
  this._rollingSize = 0;
  this._processor = null;
  this._source = null;
  this._silentGain = null;
}

/**
 * SYNCHRONOUS — call inside the tap handler, before any await.
 * Creates and unlocks the AudioContext for iOS.
 * Throws a diagnostic-rich error if the environment lacks required APIs.
 */
SonarEngine.prototype.createAndUnlock = function () {
  var AC = window["AudioContext"] || window["webkitAudioContext"];
  if (!AC) {
    throw new Error("Web Audio API missing. " + envDiagnostics());
  }
  if (!(navigator.mediaDevices && navigator.mediaDevices.getUserMedia)) {
    // Most common cause: non-secure context (self-signed cert / plain http)
    throw new Error("Microphone API missing. " + envDiagnostics());
  }

  this.audioCtx = new AC();

  // iOS unlock: resume + play a silent 1-sample buffer, all inside the gesture
  try {
    if (this.audioCtx.state === "suspended") this.audioCtx.resume();
    var buf = this.audioCtx.createBuffer(1, 1, this.audioCtx.sampleRate);
    var src = this.audioCtx.createBufferSource();
    src.buffer = buf;
    src.connect(this.audioCtx.destination);
    src.start(0);
  } catch (_) { /* unlock is best-effort */ }
};

/** Async part — safe to await; createAndUnlock() must have run first. */
SonarEngine.prototype.init = async function (backendUrl) {
  if (!this.audioCtx) this.createAndUnlock();

  // Request microphone. NOTE: do NOT pass sampleRate here — Safari can
  // reject the constraint. The server resamples whatever we send.
  this.stream = await navigator.mediaDevices.getUserMedia({
    audio: { echoCancellation: false, noiseSuppression: false, autoGainControl: false },
  });

  // iOS 17+/Safari: with the mic open, output may route to the receiver at
  // low volume. audioSession is the only knob the web has; best-effort.
  try {
    if (navigator.audioSession) navigator.audioSession.type = "play-and-record";
  } catch (_) {}

  // Load reference chirp from backend
  var resp = await fetch(backendUrl + "/chirp");
  if (!resp.ok) throw new Error("GET /chirp failed: " + resp.status);
  var data = await resp.json();
  this.chirpSampleRate = data.sample_rate;

  // Decode base64 float64 chirp — copy into an aligned buffer for Safari
  var binaryStr = atob(data.chirp_base64);
  var bytes = new Uint8Array(binaryStr.length);
  for (var i = 0; i < binaryStr.length; i++) bytes[i] = binaryStr.charCodeAt(i);
  var aligned = new ArrayBuffer(bytes.length);
  new Uint8Array(aligned).set(bytes);
  var float64 = new Float64Array(aligned);
  var float32 = new Float32Array(float64.length);
  for (var j = 0; j < float64.length; j++) float32[j] = float64[j];

  this.chirpBuffer = this.audioCtx.createBuffer(1, float32.length, this.chirpSampleRate);
  this.chirpBuffer.getChannelData(0).set(float32);

  // PERMANENT audio pipeline — never disconnect while running
  var sr = this.audioCtx.sampleRate;
  this._rollingSize = Math.ceil((this.captureS + 1.0) * sr); // 1 s headroom
  this._rollingBuffer = new Float32Array(this._rollingSize);
  this._rollingPos = 0;

  this._source = this.audioCtx.createMediaStreamSource(this.stream);
  this._processor = this.audioCtx.createScriptProcessor(4096, 1, 1);

  var self = this;
  this._processor.onaudioprocess = function (e) {
    var input = e.inputBuffer.getChannelData(0);
    for (var k = 0; k < input.length; k++) {
      self._rollingBuffer[self._rollingPos] = input[k];
      self._rollingPos = (self._rollingPos + 1) % self._rollingSize;
    }
  };

  this._source.connect(this._processor);

  // Silent gain keeps the graph pulling data without audible mic passthrough
  this._silentGain = this.audioCtx.createGain();
  this._silentGain.gain.value = 0;
  this._processor.connect(this._silentGain);
  this._silentGain.connect(this.audioCtx.destination);

  this.ready = true;
};

/** Total estimated output latency in seconds (device + manual calibration). */
SonarEngine.prototype._outputLatencyS = function () {
  var lat = 0;
  try { lat += this.audioCtx.outputLatency || 0; } catch (_) {}
  try { lat += this.audioCtx.baseLatency || 0; } catch (_) {}
  return lat + this.extraLatencyMs / 1000;
};

SonarEngine.prototype.capture = async function () {
  if (!this.ready) throw new Error("Not initialized");

  if (this.audioCtx.state === "suspended") {
    // Works if we still hold transient activation; harmless otherwise.
    try { await this.audioCtx.resume(); } catch (_) {}
    if (this.audioCtx.state === "suspended") {
      throw new Error("Audio suspended — tap the screen to resume");
    }
  }

  var latencyS = this._outputLatencyS();

  // Schedule the chirp slightly ahead so start time is deterministic
  var chirpSource = this.audioCtx.createBufferSource();
  chirpSource.buffer = this.chirpBuffer;
  chirpSource.connect(this.audioCtx.destination);
  chirpSource.start(this.audioCtx.currentTime + SCHEDULE_AHEAD_S);

  // Wait until: schedule delay + output latency + full capture window.
  // Grabbing the LAST captureS samples then aligns the window start with
  // the physical chirp emission (± mic input latency).
  var waitMs =
    SCHEDULE_AHEAD_S * 1000 + latencyS * 1000 + this.captureS * 1000 + EXTRA_MARGIN_MS;
  await new Promise(function (r) { setTimeout(r, waitMs); });

  var sr = this.audioCtx.sampleRate;
  var nSamples = Math.ceil(this.captureS * sr);
  var result = new Float32Array(nSamples);
  var readPos = (this._rollingPos - nSamples + this._rollingSize) % this._rollingSize;
  for (var i = 0; i < nSamples; i++) {
    result[i] = this._rollingBuffer[(readPos + i) % this._rollingSize];
  }

  return { audio: result, sampleRate: sr };
};

SonarEngine.prototype.predict = async function (backendUrl, audio, sampleRate, gyroPitch) {
  // Float32 payload — half the upload of float64
  var f32 = audio instanceof Float32Array ? audio : new Float32Array(audio);
  var bytes = new Uint8Array(f32.buffer, f32.byteOffset, f32.byteLength);
  var binary = "";
  var chunkSize = 8192;
  for (var i = 0; i < bytes.length; i += chunkSize) {
    var slice = bytes.subarray(i, Math.min(i + chunkSize, bytes.length));
    binary += String.fromCharCode.apply(null, slice);
  }
  var b64 = btoa(binary);

  var resp = await fetch(backendUrl + "/predict", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      audio_base64: b64,
      audio_dtype: "float32",
      sample_rate: sampleRate,
      gyroscope_pitch_deg: gyroPitch || 0,
    }),
  });

  if (!resp.ok) throw new Error("Predict failed: " + resp.status);
  return resp.json();
};

SonarEngine.prototype.destroy = function () {
  try { if (this._processor) this._processor.disconnect(); } catch (_) {}
  try { if (this._source) this._source.disconnect(); } catch (_) {}
  try { if (this._silentGain) this._silentGain.disconnect(); } catch (_) {}
  if (this.stream) {
    this.stream.getTracks().forEach(function (t) { try { t.stop(); } catch (_) {} });
  }
  try { if (this.audioCtx && this.audioCtx.state !== "closed") this.audioCtx.close(); } catch (_) {}
  this.ready = false;
};

/** Vibration — Android only; iOS Safari has no vibration API. */
export function vibrate(pattern) {
  if (!navigator.vibrate) return;
  var patterns = {
    none: [], single_pulse: [100], double_pulse: [80, 60, 80],
    double_pulse_slow: [120, 100, 120], continuous_low: [300],
    continuous_high: [50, 30, 50, 30, 50, 30, 50],
    stair_alert: [200, 100, 200, 100, 400],
  };
  navigator.vibrate(patterns[pattern] || [100]);
}

export function GyroReader() {
  this.pitch = 0;
  this._handler = null;
}

/**
 * Must be CALLED (not awaited-after-other-awaits) inside the user gesture:
 * DeviceOrientationEvent.requestPermission() requires transient activation
 * on iOS. Kick it off first in the tap handler, await the promise later.
 */
GyroReader.prototype.start = function () {
  var self = this;
  this._handler = function (e) { self.pitch = e.beta || 0; };

  var permissionPromise = Promise.resolve("granted");
  try {
    if (typeof DeviceOrientationEvent !== "undefined" &&
        typeof DeviceOrientationEvent.requestPermission === "function") {
      permissionPromise = DeviceOrientationEvent.requestPermission();
    }
  } catch (_) {}

  return permissionPromise
    .then(function (perm) {
      if (perm !== "granted") return false;
      window.addEventListener("deviceorientation", self._handler);
      return true;
    })
    .catch(function () {
      // Non-iOS or permission API absent: just listen
      window.addEventListener("deviceorientation", self._handler);
      return true;
    });
};

GyroReader.prototype.stop = function () {
  if (this._handler) window.removeEventListener("deviceorientation", this._handler);
};
