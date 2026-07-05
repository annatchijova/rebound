/**
 * Audio feedback for eyes-free use.
 *
 * Two channels:
 *   1. Earcons — instant oscillator patterns. Redesigned so each class has a
 *      DIFFERENT RHYTHM + MELODIC CONTOUR, not just a different pitch.
 *      Pitch alone is indistinguishable on phone speakers; rhythm and
 *      direction (rising vs falling) are what people actually perceive.
 *   2. Speech — speechSynthesis TTS (built into iOS and Android, offline,
 *      free). Speaks class + distance, throttled so it doesn't chatter.
 *
 * Earcon design language:
 *   open_space  — one long, soft, low tone. "Nothing here, relax."
 *   corridor    — two identical medium tones. "Steady path."
 *   doorway     — RISING interval (ding-DING↑). "Opening / invitation."
 *   corner      — FALLING interval (DING-dong↓). "Turn coming."
 *   nearby_wall — parking-sensor beeps: rate increases as distance shrinks.
 *                 Universally understood from cars.
 *   stairs      — two-pitch alternating siren. Unmistakable alarm.
 */

export function AudioFeedback(audioCtx) {
  this.ctx = audioCtx;
}

AudioFeedback.prototype._tone = function (freq, start, dur, vol, type) {
  var ctx = this.ctx;
  var osc = ctx.createOscillator();
  var gain = ctx.createGain();
  osc.type = type || "sine";
  osc.frequency.value = freq;
  gain.gain.setValueAtTime(0.0001, start);
  gain.gain.exponentialRampToValueAtTime(vol, start + 0.01);
  gain.gain.exponentialRampToValueAtTime(0.0001, start + dur);
  osc.connect(gain);
  gain.connect(ctx.destination);
  osc.start(start);
  osc.stop(start + dur + 0.02);
};

AudioFeedback.prototype.play = function (className, distance) {
  var now = this.ctx.currentTime;
  var d = typeof distance === "number" ? distance : 3.0;

  switch (className) {
    case "open_space":
      // One long soft low tone
      this._tone(330, now, 0.35, 0.12, "sine");
      break;

    case "corridor":
      // Two identical medium tones — steady rhythm
      this._tone(520, now, 0.14, 0.22, "sine");
      this._tone(520, now + 0.22, 0.14, 0.22, "sine");
      break;

    case "doorway":
      // Rising interval — perfect fifth up
      this._tone(523, now, 0.12, 0.25, "sine");
      this._tone(784, now + 0.15, 0.18, 0.3, "sine");
      break;

    case "corner":
      // Falling interval — mirror of doorway
      this._tone(660, now, 0.12, 0.28, "triangle");
      this._tone(440, now + 0.15, 0.18, 0.28, "triangle");
      break;

    case "nearby_wall": {
      // Parking sensor: closer = faster beeps. 3 m → slow, <0.7 m → near-continuous
      var gap = Math.max(0.05, Math.min(0.3, d * 0.09));
      var beeps = d < 0.8 ? 6 : d < 1.5 ? 4 : 3;
      for (var i = 0; i < beeps; i++) {
        this._tone(1000, now + i * (0.06 + gap), 0.05, 0.45, "square");
      }
      break;
    }

    case "stairs": {
      // Two-pitch alternating siren — nothing else in the set sounds like this
      var freqs = [880, 587, 880, 587, 880];
      for (var s = 0; s < freqs.length; s++) {
        this._tone(freqs[s], now + s * 0.11, 0.09, 0.5, "sawtooth");
      }
      break;
    }

    default:
      this._tone(440, now, 0.15, 0.2, "sine");
  }
};

// Startup confirmation — rising major arpeggio, "system ready"
AudioFeedback.prototype.playReady = function () {
  var now = this.ctx.currentTime;
  var freqs = [440, 554, 659];
  for (var i = 0; i < freqs.length; i++) {
    this._tone(freqs[i], now + i * 0.12, 0.1, 0.2, "sine");
  }
};

// Error — low buzz, clearly negative
AudioFeedback.prototype.playError = function () {
  var now = this.ctx.currentTime;
  this._tone(180, now, 0.3, 0.25, "square");
};

/* ------------------------------------------------------------------ */
/* Speech — TTS announcements                                          */
/* ------------------------------------------------------------------ */

var PHRASES = {
  es: {
    open_space: "Espacio abierto",
    nearby_wall: "Pared a {d} metros",
    doorway: "Puerta a {d} metros",
    corner: "Esquina a {d} metros",
    corridor: "Pasillo",
    stairs_up: "Atención: escaleras hacia arriba",
    stairs_down: "Atención: escaleras hacia abajo",
    stairs: "Atención: escaleras a {d} metros",
    ready: "Sonar activo",
    error: "Error de escaneo",
  },
  en: {
    open_space: "Open space",
    nearby_wall: "Wall at {d} meters",
    doorway: "Doorway at {d} meters",
    corner: "Corner at {d} meters",
    corridor: "Corridor",
    stairs_up: "Caution: stairs going up",
    stairs_down: "Caution: stairs going down",
    stairs: "Caution: stairs at {d} meters",
    ready: "Sonar active",
    error: "Scan error",
  },
};

export function Speech(lang) {
  var nav = (navigator.language || "en").slice(0, 2);
  this.lang = lang === "es" || lang === "en" ? lang : (nav === "es" ? "es" : "en");
  this.supported = "speechSynthesis" in window;
  this.enabled = this.supported;
  this._lastText = "";
  this._lastAt = 0;
}

/** Call synchronously inside a user gesture — unlocks TTS on iOS. */
Speech.prototype.unlock = function () {
  if (!this.supported) return;
  try {
    window.speechSynthesis.cancel();
    var u = new SpeechSynthesisUtterance(" ");
    u.volume = 0;
    window.speechSynthesis.speak(u);
  } catch (_) {}
};

/**
 * Build the announcement for a prediction. Distance is rounded to 0.5 m so
 * the text (and thus the throttled speech) stays stable between scans.
 */
Speech.prototype.phraseFor = function (pred) {
  var P = PHRASES[this.lang];
  var d = Math.max(0.5, Math.round(pred.distance_m * 2) / 2);
  var dStr = (d % 1 === 0 ? String(d) : d.toFixed(1));

  if (pred.class_name === "stairs") {
    if (pred.stair_direction === "up") return P.stairs_up;
    if (pred.stair_direction === "down") return P.stairs_down;
    return P.stairs.replace("{d}", dStr);
  }
  var tpl = P[pred.class_name] || pred.class_name;
  return tpl.replace("{d}", dStr);
};

Speech.prototype.say = function (text, opts) {
  if (!this.enabled || !this.supported || !text) return;
  opts = opts || {};
  var now = Date.now();

  // Throttle: never repeat the same text within 5 s; never speak more often
  // than every 2 s — unless it's urgent (stairs).
  if (!opts.urgent) {
    if (text === this._lastText && now - this._lastAt < 5000) return;
    if (now - this._lastAt < 2000) return;
  }

  try {
    window.speechSynthesis.cancel(); // latest info wins
    var u = new SpeechSynthesisUtterance(text);
    u.lang = this.lang === "es" ? "es-ES" : "en-US";
    u.rate = 1.15;
    u.volume = 1.0;
    window.speechSynthesis.speak(u);
    this._lastText = text;
    this._lastAt = now;
  } catch (_) {}
};

Speech.prototype.ready = function () { this.say(PHRASES[this.lang].ready, { urgent: true }); };
Speech.prototype.stop = function () {
  if (this.supported) { try { window.speechSynthesis.cancel(); } catch (_) {} }
};
