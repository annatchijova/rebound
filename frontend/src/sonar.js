/**
 * REBOUND Sonar Engine — Web Audio API
 *
 * Keeps microphone connected permanently to prevent AudioContext suspension.
 * Chirp is played and audio is grabbed from a rolling buffer on each scan.
 */

var CAPTURE_DURATION_S = 0.15;

export function SonarEngine() {
  this.audioCtx = null;
  this.chirpBuffer = null;
  this.chirpSampleRate = 44100;
  this.stream = null;
  this.ready = false;

  // Rolling buffer — always recording
  this._rollingBuffer = null;
  this._rollingPos = 0;
  this._rollingSize = 0;
  this._processor = null;
  this._source = null;
  this._silentGain = null;
}

SonarEngine.prototype.init = async function(backendUrl) {
  // Create AudioContext — bracket notation survives minification
  var AC = window["AudioContext"] || window["webkitAudioContext"];
  if (!AC) throw new Error("AudioContext not available");
  this.audioCtx = new AC();

  // Request microphone
  this.stream = await navigator.mediaDevices.getUserMedia({
    audio: { echoCancellation: false, noiseSuppression: false, autoGainControl: false }
  });

  // Load chirp from backend
  var resp = await fetch(backendUrl + "/chirp");
  var data = await resp.json();
  this.chirpSampleRate = data.sample_rate;

  // Decode base64 chirp — aligned buffer for Safari
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

  // Set up PERMANENT audio pipeline — never disconnect
  var sr = this.audioCtx.sampleRate;
  this._rollingSize = Math.ceil(CAPTURE_DURATION_S * sr) + 4096;
  this._rollingBuffer = new Float32Array(this._rollingSize);
  this._rollingPos = 0;

  this._source = this.audioCtx.createMediaStreamSource(this.stream);
  this._processor = this.audioCtx.createScriptProcessor(4096, 1, 1);

  var self = this;
  this._processor.onaudioprocess = function(e) {
    var input = e.inputBuffer.getChannelData(0);
    for (var k = 0; k < input.length; k++) {
      self._rollingBuffer[self._rollingPos] = input[k];
      self._rollingPos = (self._rollingPos + 1) % self._rollingSize;
    }
  };

  this._source.connect(this._processor);

  // Silent gain node keeps AudioContext alive without audible output from mic
  this._silentGain = this.audioCtx.createGain();
  this._silentGain.gain.value = 0;
  this._processor.connect(this._silentGain);
  this._silentGain.connect(this.audioCtx.destination);

  this.ready = true;
};

SonarEngine.prototype.capture = async function() {
  if (!this.ready) throw new Error("Not initialized");

  if (this.audioCtx.state === "suspended") {
    await this.audioCtx.resume();
  }

  // Play chirp
  var chirpSource = this.audioCtx.createBufferSource();
  chirpSource.buffer = this.chirpBuffer;
  chirpSource.connect(this.audioCtx.destination);
  chirpSource.start(0);

  // Wait for capture duration
  var captureMs = Math.ceil(CAPTURE_DURATION_S * 1000) + 50;
  await new Promise(function(r) { setTimeout(r, captureMs); });

  // Grab audio from rolling buffer
  var sr = this.audioCtx.sampleRate;
  var nSamples = Math.ceil(CAPTURE_DURATION_S * sr);
  var result = new Float32Array(nSamples);

  var readPos = (this._rollingPos - nSamples + this._rollingSize) % this._rollingSize;
  for (var i = 0; i < nSamples; i++) {
    result[i] = this._rollingBuffer[(readPos + i) % this._rollingSize];
    }

  // Convert to Float64
  var f64 = new Float64Array(nSamples);
  for (var j = 0; j < nSamples; j++) f64[j] = result[j];

  return { audio: f64, sampleRate: sr };
};

SonarEngine.prototype.predict = async function(backendUrl, audio, sampleRate, gyroPitch) {
  var bytes = new Uint8Array(audio.buffer);
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
      sample_rate: sampleRate,
      gyroscope_pitch_deg: gyroPitch || 0,
    }),
  });

  if (!resp.ok) throw new Error("Predict failed: " + resp.status);
  return resp.json();
};

SonarEngine.prototype.destroy = function() {
  if (this._processor) this._processor.disconnect();
  if (this._source) this._source.disconnect();
  if (this._silentGain) this._silentGain.disconnect();
  if (this.stream) {
    this.stream.getTracks().forEach(function(t) { t.stop(); });
  }
  if (this.audioCtx) this.audioCtx.close();
};

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

GyroReader.prototype.start = async function() {
  var self = this;
  this._handler = function(e) { self.pitch = e.beta || 0; };
  try {
    if (typeof DeviceOrientationEvent !== "undefined" &&
        typeof DeviceOrientationEvent.requestPermission === "function") {
      var perm = await DeviceOrientationEvent.requestPermission();
      if (perm !== "granted") return false;
    }
  } catch (_) {}
  window.addEventListener("deviceorientation", this._handler);
  return true;
};

GyroReader.prototype.stop = function() {
  if (this._handler) window.removeEventListener("deviceorientation", this._handler);
};
