/**
 * REBOUND Sonar Engine — Web Audio API
 *
 * Handles chirp playback through speaker + simultaneous mic recording.
 * Returns the captured audio buffer for server-side processing.
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

  /**
   * Initialize: request mic permission + load chirp from server.
   */
  async init(backendUrl) {
    this.audioCtx = new AudioContext();

    // Request microphone
    this.stream = await navigator.mediaDevices.getUserMedia({
      audio: {
        echoCancellation: false,
        noiseSuppression: false,
        autoGainControl: false,
        sampleRate: 44100,
      },
    });

    // Load chirp from backend
    const resp = await fetch(`${backendUrl}/chirp`);
    const data = await resp.json();
    this.chirpSampleRate = data.sample_rate;

    // Decode base64 float64 chirp into AudioBuffer
    const raw = Uint8Array.from(atob(data.chirp_base64), (c) => c.charCodeAt(0));
    const float64 = new Float64Array(raw.buffer);
    const float32 = new Float32Array(float64.length);
    for (let i = 0; i < float64.length; i++) float32[i] = float64[i];

    this.chirpBuffer = this.audioCtx.createBuffer(1, float32.length, this.chirpSampleRate);
    this.chirpBuffer.getChannelData(0).set(float32);

    this.ready = true;
  }

  /**
   * Emit chirp + record echo. Returns Float64Array of captured audio.
   */
  async capture() {
    if (!this.ready) throw new Error("SonarEngine not initialized");

    // Resume context if suspended (mobile autoplay policy)
    if (this.audioCtx.state === "suspended") {
      await this.audioCtx.resume();
    }

    const sampleRate = this.audioCtx.sampleRate;
    const captureSamples = Math.ceil(CAPTURE_DURATION_S * sampleRate);

    return new Promise((resolve, reject) => {
      // Set up recorder via ScriptProcessorNode (wider browser support)
      const source = this.audioCtx.createMediaStreamSource(this.stream);
      const processor = this.audioCtx.createScriptProcessor(4096, 1, 1);
      const chunks = [];
      let samplesCollected = 0;

      processor.onaudioprocess = (e) => {
        const input = e.inputBuffer.getChannelData(0);
        chunks.push(new Float32Array(input));
        samplesCollected += input.length;
        if (samplesCollected >= captureSamples) {
          // Done recording
          source.disconnect();
          processor.disconnect();

          // Merge chunks
          const total = new Float32Array(samplesCollected);
          let offset = 0;
          for (const chunk of chunks) {
            total.set(chunk, offset);
            offset += chunk.length;
          }

          // Trim to exact capture length
          const trimmed = total.slice(0, captureSamples);

          // Convert to Float64Array for server
          const f64 = new Float64Array(trimmed.length);
          for (let i = 0; i < trimmed.length; i++) f64[i] = trimmed[i];

          resolve({ audio: f64, sampleRate });
        }
      };

      source.connect(processor);
      processor.connect(this.audioCtx.destination);

      // Play chirp through speaker
      const chirpSource = this.audioCtx.createBufferSource();
      chirpSource.buffer = this.chirpBuffer;
      chirpSource.connect(this.audioCtx.destination);
      chirpSource.start();
    });
  }

  /**
   * Send captured audio to POST /predict.
   */
  async predict(backendUrl, audio, sampleRate, gyroPitch = 0) {
    // Encode as base64
    const bytes = new Uint8Array(audio.buffer);
    let binary = "";
    for (let i = 0; i < bytes.length; i++) binary += String.fromCharCode(bytes[i]);
    const b64 = btoa(binary);

    const resp = await fetch(`${backendUrl}/predict`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        audio_base64: b64,
        sample_rate: sampleRate,
        gyroscope_pitch_deg: gyroPitch,
      }),
    });

    if (!resp.ok) throw new Error(`Predict failed: ${resp.status}`);
    return resp.json();
  }

  destroy() {
    if (this.stream) {
      this.stream.getTracks().forEach((t) => t.stop());
    }
    if (this.audioCtx) {
      this.audioCtx.close();
    }
  }
}

/**
 * Haptic feedback via navigator.vibrate().
 * Maps REBOUND haptic patterns to vibration sequences.
 */
export function vibrate(pattern) {
  if (!navigator.vibrate) return;

  const patterns = {
    none: [],
    single_pulse: [100],
    double_pulse: [80, 60, 80],
    double_pulse_slow: [120, 100, 120],
    continuous_low: [300],
    continuous_high: [50, 30, 50, 30, 50, 30, 50],
    stair_alert: [200, 100, 200, 100, 400],
  };

  const seq = patterns[pattern] || [100];
  navigator.vibrate(seq);
}

/**
 * Gyroscope pitch reading via DeviceOrientationEvent.
 * Returns current pitch angle in degrees.
 */
export class GyroReader {
  constructor() {
    this.pitch = 0;
    this._handler = (e) => {
      // beta = front-back tilt (-180 to 180)
      this.pitch = e.beta || 0;
    };
  }

  async start() {
    // iOS 13+ requires permission
    if (typeof DeviceOrientationEvent.requestPermission === "function") {
      const perm = await DeviceOrientationEvent.requestPermission();
      if (perm !== "granted") return false;
    }
    window.addEventListener("deviceorientation", this._handler);
    return true;
  }

  stop() {
    window.removeEventListener("deviceorientation", this._handler);
  }
}
