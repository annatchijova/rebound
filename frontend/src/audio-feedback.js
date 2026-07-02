/**
 * Audio feedback via oscillator tones — works everywhere, no TTS dependency.
 *
 * Each class has a distinct sound pattern. Distance modulates repetition rate:
 * closer = faster beeps.
 */

export function AudioFeedback(audioCtx) {
  this.ctx = audioCtx;
}

// Sound patterns per class
var PATTERNS = {
  open_space:  { freq: 440, type: "sine",     beeps: 1, dur: 0.15, vol: 0.15 },  // gentle
  nearby_wall: { freq: 980, type: "square",   beeps: 4, dur: 0.06, vol: 0.5  },  // LOUD rapid
  doorway:     { freq: 550, type: "sine",     beeps: 2, dur: 0.12, vol: 0.25 },  // two-tone
  corner:      { freq: 660, type: "triangle", beeps: 2, dur: 0.10, vol: 0.3  },  // medium
  corridor:    { freq: 500, type: "sine",     beeps: 1, dur: 0.20, vol: 0.2  },  // steady
  stairs:      { freq: 880, type: "sawtooth", beeps: 5, dur: 0.05, vol: 0.5  },  // ALARM
};

AudioFeedback.prototype.play = function(className, distance) {
  var p = PATTERNS[className] || PATTERNS.open_space;
  var ctx = this.ctx;

  // Closer = higher pitch shift
  var pitchMult = distance < 1.0 ? 1.3 : distance < 2.0 ? 1.1 : 1.0;
  // Closer = shorter gap between beeps
  var gap = distance < 1.0 ? 0.06 : distance < 2.0 ? 0.10 : 0.15;

  var now = ctx.currentTime;
  for (var i = 0; i < p.beeps; i++) {
    var start = now + i * (p.dur + gap);
    var osc = ctx.createOscillator();
    var gain = ctx.createGain();
    osc.type = p.type;
    osc.frequency.value = p.freq * pitchMult;
    gain.gain.setValueAtTime(p.vol, start);
    gain.gain.exponentialRampToValueAtTime(0.01, start + p.dur);
    osc.connect(gain);
    gain.connect(ctx.destination);
    osc.start(start);
    osc.stop(start + p.dur + 0.01);
  }
};

// Startup confirmation beep
AudioFeedback.prototype.playReady = function() {
  var ctx = this.ctx;
  var now = ctx.currentTime;
  [440, 554, 659].forEach(function(freq, i) {
    var osc = ctx.createOscillator();
    var gain = ctx.createGain();
    osc.type = "sine";
    osc.frequency.value = freq;
    gain.gain.setValueAtTime(0.2, now + i * 0.12);
    gain.gain.exponentialRampToValueAtTime(0.01, now + i * 0.12 + 0.10);
    osc.connect(gain);
    gain.connect(ctx.destination);
    osc.start(now + i * 0.12);
    osc.stop(now + i * 0.12 + 0.11);
  });
};

// Error beep
AudioFeedback.prototype.playError = function() {
  var ctx = this.ctx;
  var osc = ctx.createOscillator();
  var gain = ctx.createGain();
  osc.type = "square";
  osc.frequency.value = 200;
  gain.gain.setValueAtTime(0.25, ctx.currentTime);
  gain.gain.exponentialRampToValueAtTime(0.01, ctx.currentTime + 0.3);
  osc.connect(gain);
  gain.connect(ctx.destination);
  osc.start();
  osc.stop(ctx.currentTime + 0.31);
};
