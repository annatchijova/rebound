/**
 * REBOUND Sonar Engine — Web Audio API
 *
 * Handles chirp playback through speaker + simultaneous mic recording.
 * Compatible with Chrome (Android) and Safari (iOS).
 */

const CAPTURE_DURATION_S = 0.15;

export class SonarEngine {
  constructor() {
    this.audioCtx = null;
    this.chirpBuffer = null;
    this.chirpSampleRate = 44100;
    this.stream = null;
    this.ready = false;
  }

  async init(backendUrl) {
    // Access AudioContext via bracket notation — minifier cannot rename string keys
    var AC = window["AudioContext"] || window["webkitAudioContext"];
    if (!AC) throw new Error("AudioContext not available");
    this.audioCtx = new AC();

    // Request microphone — don't constrain sampleRate (Safari ignores it)
    this.stream = await navigator.mediaDevices.getUserMedia({
      audio: {
        echoCancellation: false,
        noiseSuppression: false,
        autoGainControl: false,
      },
    });

    // Load chirp from backend
    const resp = await fetch(backendUrl + "/chirp");
    const data = await resp.json();
    this.chirpSampleRate = data.sample_rate;

    // Decode base64 chirp — Safari-safe (no Float64Array alignment issues)
    const binaryStr = atob(data.chirp_base64);
    const len = binaryStr.length;
    const bytes = new Uint8Array(len);
    for (let i = 0; i < len; i++) bytes[i] = binaryStr.charCodeAt(i);

    // Copy to aligned buffer for Float64Array
    const aligned = new ArrayBuffer(bytes.length);
    new Uint8Array(aligned).set(bytes);
    const float64 = new Float64Array(aligned);

    const float32 = new Float32Array(float64.length);
    for (let i = 0; i < float64.length; i++) float32[i] = float64[i];

    this.chirpBuffer = this.audioCtx.createBuffer(1, float32.length, this.chirpSampleRate);
    this.chirpBuffer.getChannelData(0).set(float32);

    this.ready = true;
  }

  async capture() {
    if (!this.ready) throw new Error("SonarEngine not initialized");

    if (this.audioCtx.state === "suspended") {
      await this.audioCtx.resume();
    }

    var sampleRate = this.audioCtx.sampleRate;
    var captureSamples = Math.ceil(CAPTURE_DURATION_S * sampleRate);
    var audioCtx = this.audioCtx;
    var stream = this.stream;
    var chirpBuffer = this.chirpBuffer;

    return new Promise(function (resolve) {
      var source = audioCtx.createMediaStreamSource(stream);
      var bufferSize = 4096;
      var processor = audioCtx.createScriptProcessor(bufferSize, 1, 1);
      var chunks = [];
      var samplesCollected = 0;

      processor.onaudioprocess = function (e) {
        var input = e.inputBuffer.getChannelData(0);
        var copy = new Float32Array(input.length);
        copy.set(input);
        chunks.push(copy);
        samplesCollected += input.length;

        if (samplesCollected >= captureSamples) {
          source.disconnect();
          processor.disconnect();

          var total = new Float32Array(samplesCollected);
          var offset = 0;
          for (var i = 0; i < chunks.length; i++) {
            total.set(chunks[i], offset);
            offset += chunks[i].length;
          }

          var trimmed = total.slice(0, captureSamples);
          var f64 = new Float64Array(trimmed.length);
          for (var j = 0; j < trimmed.length; j++) f64[j] = trimmed[j];

          resolve({ audio: f64, sampleRate: sampleRate });
        }
      };

      source.connect(processor);
      processor.connect(audioCtx.destination);

      // Play chirp
      var chirpSource = audioCtx.createBufferSource();
      chirpSource.buffer = chirpBuffer;
      chirpSource.connect(audioCtx.destination);
      chirpSource.start(0);
    });
  }

  async predict(backendUrl, audio, sampleRate, gyroPitch) {
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
  }

  destroy() {
    if (this.stream) {
      this.stream.getTracks().forEach(function (t) { t.stop(); });
    }
    if (this.audioCtx) {
      this.audioCtx.close();
    }
  }
}

export function vibrate(pattern) {
  if (!navigator.vibrate) return;

  var patterns = {
    none: [],
    single_pulse: [100],
    double_pulse: [80, 60, 80],
    double_pulse_slow: [120, 100, 120],
    continuous_low: [300],
    continuous_high: [50, 30, 50, 30, 50, 30, 50],
    stair_alert: [200, 100, 200, 100, 400],
  };

  navigator.vibrate(patterns[pattern] || [100]);
}

export class GyroReader {
  constructor() {
    this.pitch = 0;
    this._handler = null;
  }

  async start() {
    var self = this;
    this._handler = function (e) {
      self.pitch = e.beta || 0;
    };

    try {
      if (typeof DeviceOrientationEvent !== "undefined" &&
          typeof DeviceOrientationEvent.requestPermission === "function") {
        var perm = await DeviceOrientationEvent.requestPermission();
        if (perm !== "granted") return false;
      }
    } catch (e) {
      // Permission API not available — continue without gyro
    }

    window.addEventListener("deviceorientation", this._handler);
    return true;
  }

  stop() {
    if (this._handler) {
      window.removeEventListener("deviceorientation", this._handler);
    }
  }
}
