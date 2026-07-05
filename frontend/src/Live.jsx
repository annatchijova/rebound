import { useState, useRef, useEffect } from "react";
import { SonarEngine, vibrate, GyroReader, envDiagnostics } from "./sonar.js";
import { AudioFeedback, Speech } from "./audio-feedback.js";

var CLASS_LABELS = {
  open_space: "Open space", nearby_wall: "Nearby wall", doorway: "Doorway",
  corner: "Corner", corridor: "Corridor", stairs: "Stairs",
};

var CLASS_COLORS = {
  open_space: "#00D4FF", nearby_wall: "#F59E0B", doorway: "#10B981",
  corner: "#F97316", corridor: "#A78BFA", stairs: "#EF4444",
};

var HAPTIC_FOR_CLASS = {
  open_space: "none", nearby_wall: "continuous_high", doorway: "double_pulse_slow",
  corner: "continuous_low", corridor: "single_pulse", stairs: "stair_alert",
};

// /process (Qwen agent) is expensive — call it on class change or every N scans
var PROCESS_EVERY_N_SCANS = 8;

// Visually hidden but visible to screen readers (VoiceOver / TalkBack)
var SR_ONLY = {
  position: "absolute", width: 1, height: 1, padding: 0, margin: -1,
  overflow: "hidden", clip: "rect(0 0 0 0)", whiteSpace: "nowrap", border: 0,
};

export default function Live({ backendUrl }) {
  var [phase, setPhase] = useState("start");
  var [error, setError] = useState("");
  var [result, setResult] = useState(null);
  var [scanCount, setScanCount] = useState(0);
  var [agentResult, setAgentResult] = useState(null);
  var [debug, setDebug] = useState("");
  var [announcement, setAnnouncement] = useState("");
  var [urgentAnnouncement, setUrgentAnnouncement] = useState("");

  var engineRef = useRef(null);
  var gyroRef = useRef(null);
  var feedbackRef = useRef(null);
  var speechRef = useRef(null);
  var initRef = useRef(false);
  var runningRef = useRef(false);
  var scanningRef = useRef(false);   // prevents overlapping scans
  var lastClassRef = useRef("");
  var scanCountRef = useRef(0);
  var userIdRef = useRef(null);

  // URL params: ?lang=es|en ?voice=0 ?user=judge1 ?lat=120 (ms) ?token=SECRET
  if (userIdRef.current === null) {
    var params = new URLSearchParams(window.location.search);
    userIdRef.current = {
      lang: params.get("lang") || "",
      voice: params.get("voice") !== "0",
      user: (params.get("user") || "mobile_user").replace(/[^a-zA-Z0-9_\-]/g, "").slice(0, 64) || "mobile_user",
      lat: parseFloat(params.get("lat")) || 0,
      token: params.get("token") || "",
    };
  }

  // Single scan cycle
  async function doScan() {
    if (!engineRef.current || scanningRef.current) return;
    scanningRef.current = true;
    try {
      var t0 = Date.now();
      var cap = await engineRef.current.capture();
      var pitch = gyroRef.current ? gyroRef.current.pitch : 0;
      var pred = await engineRef.current.predict(
        backendUrl, cap.audio, cap.sampleRate, pitch, userIdRef.current.token
      );
      var ms = Date.now() - t0;

      setResult(pred);
      scanCountRef.current += 1;
      setScanCount(scanCountRef.current);

      var classChanged = pred.class_name !== lastClassRef.current;
      lastClassRef.current = pred.class_name;

      // 1. Earcon — instant
      if (feedbackRef.current) feedbackRef.current.play(pred.class_name, pred.distance_m);
      // 2. Vibration — Android only (iOS has no vibrate API)
      vibrate(HAPTIC_FOR_CLASS[pred.class_name] || "single_pulse");
      // 3. Speech + screen-reader live region
      if (speechRef.current) {
        var phrase = speechRef.current.phraseFor(pred);
        var isStairs = pred.class_name === "stairs";
        speechRef.current.say(phrase, { urgent: isStairs && classChanged });
        if (isStairs) setUrgentAnnouncement(phrase);
        else setAnnouncement(phrase);
      }

      setDebug(
        pred.class_name + " " + Math.round(pred.confidence * 100) + "% · SNR " +
        pred.snr_db + " dB · " + ms + " ms"
      );

      // Memory agent — throttled: class change or every N scans
      if (classChanged || scanCountRef.current % PROCESS_EVERY_N_SCANS === 0) {
        var processHeaders = { "Content-Type": "application/json" };
        if (userIdRef.current.token) processHeaders["X-API-Token"] = userIdRef.current.token;
        fetch(backendUrl + "/process", {
          method: "POST",
          headers: processHeaders,
          body: JSON.stringify({
            user_id: userIdRef.current.user,
            prediction: {
              class_name: pred.class_name,
              confidence: pred.confidence,
              distance_m: pred.distance_m,
            },
            features_summary: pred.features_summary,
            user_action: "advance",
            session_id: 1,
          }),
        })
          .then(function (r) { return r.ok ? r.json() : null; })
          .then(function (data) { if (data) setAgentResult(data); })
          .catch(function () {});
      }
    } catch (e) {
      console.error("Scan error:", e);
      if (feedbackRef.current) feedbackRef.current.playError();
      setDebug("scan error: " + String(e && e.message ? e.message : e));
    } finally {
      scanningRef.current = false;
    }
  }

  // Auto-scan loop
  useEffect(function () {
    if (!runningRef.current) return;
    var cancelled = false;

    async function loop() {
      while (runningRef.current && !cancelled) {
        await doScan();
        if (!runningRef.current || cancelled) break;
        await new Promise(function (r) { setTimeout(r, 150); });
      }
    }
    loop();

    return function () { cancelled = true; };
  }, [phase === "running"]);

  /**
   * Start — everything permission-related happens SYNCHRONOUSLY at the top,
   * inside the tap's transient activation:
   *   1. createAndUnlock() — AudioContext resume + silent buffer (iOS unlock)
   *   2. speech.unlock()   — speechSynthesis unlock (iOS)
   *   3. gyro.start()      — DeviceOrientationEvent.requestPermission()
   * Only THEN do we await getUserMedia / fetch.
   */
  async function handleStart() {
    if (initRef.current) return;
    initRef.current = true;
    setPhase("init");
    setError("");

    var engine = new SonarEngine();
    engine.extraLatencyMs = userIdRef.current.lat;
    var gyro = new GyroReader();
    var speech = new Speech(userIdRef.current.lang);
    speech.enabled = userIdRef.current.voice && speech.supported;

    var gyroPromise = null;
    try {
      // --- synchronous, inside the gesture ---
      engine.createAndUnlock();
      speech.unlock();
      gyroPromise = gyro.start(); // fires the iOS permission prompt NOW
      // --- async from here on ---
      await engine.init(backendUrl);

      engineRef.current = engine;
      gyroRef.current = gyro;
      speechRef.current = speech;

      feedbackRef.current = new AudioFeedback(engine.audioCtx);
      feedbackRef.current.playReady();
      speech.ready();
      setAnnouncement("Sonar active. Scanning.");

      if (gyroPromise) gyroPromise.catch(function () {});

      try { if (navigator.wakeLock) await navigator.wakeLock.request("screen"); } catch (_) {}

      runningRef.current = true;
      setPhase("running");
    } catch (e) {
      try { engine.destroy(); } catch (_) {}
      initRef.current = false;
      var msg = String(e && e.message ? e.message : e);
      // Always append environment diagnostics — lets us debug judges' phones
      if (msg.indexOf("secureContext") === -1) msg += " — " + envDiagnostics();
      setError(msg);
      setAnnouncement("Error: " + msg);
      setPhase("error");
    }
  }

  // Tap: start / retry / force an extra scan
  function handleTap() {
    if (phase === "start" || phase === "error") { handleStart(); return; }
    if (phase === "running" && engineRef.current) {
      // Also re-unlocks a suspended context, since taps carry activation
      if (engineRef.current.audioCtx && engineRef.current.audioCtx.state === "suspended") {
        try { engineRef.current.audioCtx.resume(); } catch (_) {}
      }
      doScan();
    }
  }

  // Cleanup
  useEffect(function () {
    return function () {
      runningRef.current = false;
      if (engineRef.current) engineRef.current.destroy();
      if (gyroRef.current) gyroRef.current.stop();
      if (speechRef.current) speechRef.current.stop();
    };
  }, []);

  var color = result ? (CLASS_COLORS[result.class_name] || "#00D4FF") : "#00D4FF";

  return (
    <div
      onClick={handleTap}
      onTouchEnd={function (e) { e.preventDefault(); handleTap(); }}
      role="button"
      aria-label={
        phase === "start" ? "Tap to start sonar" :
        phase === "error" ? "Error. Tap to retry" : "Sonar running. Tap to scan now"
      }
      style={{
        background: "#0A0E1A", color: "#E2E8F0", minHeight: "100vh", width: "100%",
        display: "flex", flexDirection: "column", alignItems: "center",
        justifyContent: "center", fontFamily: "monospace",
        cursor: "pointer", userSelect: "none", WebkitUserSelect: "none",
        WebkitTapHighlightColor: "transparent", padding: 20,
      }}
    >
      {/* Screen-reader live regions — VoiceOver/TalkBack read these aloud */}
      <div aria-live="polite" role="status" style={SR_ONLY}>{announcement}</div>
      <div aria-live="assertive" role="alert" style={SR_ONLY}>{urgentAnnouncement}</div>

      <div style={{ position: "fixed", top: 12, left: 0, right: 0, textAlign: "center" }}>
        <span style={{ fontSize: 16, fontWeight: 700, color: "#00D4FF44", letterSpacing: 3 }}>REBOUND</span>
      </div>

      {phase === "start" && (
        <>
          <div style={{ fontSize: 28, fontWeight: 700, color: "#00D4FF", letterSpacing: 3, marginBottom: 32 }}>
            REBOUND
          </div>
          <div style={{
            width: 200, height: 200, borderRadius: "50%", background: "#00D4FF",
            display: "flex", alignItems: "center", justifyContent: "center",
            boxShadow: "0 0 80px #00D4FF44",
          }}>
            <span style={{ color: "#0A0E1A", fontSize: 18, fontWeight: 700, textAlign: "center", padding: 20 }}>
              TAP TO START
            </span>
          </div>
        </>
      )}

      {phase === "init" && (
        <div style={{ fontSize: 18, color: "#64748B" }}>Starting...</div>
      )}

      {phase === "error" && (
        <div style={{ textAlign: "center", padding: 24 }}>
          <div style={{ color: "#EF4444", fontSize: 14, marginBottom: 20, wordBreak: "break-word" }}>{error}</div>
          <div style={{ fontSize: 16, color: "#F59E0B" }}>Tap to retry</div>
        </div>
      )}

      {(phase === "running" || phase === "stopped") && (
        <>
          {result ? (
            <div style={{ textAlign: "center", width: "100%", maxWidth: 360 }}>
              <div style={{ fontSize: 80, fontWeight: 700, color: color, lineHeight: 1 }}>
                {result.distance_m.toFixed(1)}
              </div>
              <div style={{ fontSize: 14, color: "#64748B", marginBottom: 12 }}>meters</div>
              <div style={{ fontSize: 32, fontWeight: 700, color: color, marginBottom: 16 }}>
                {CLASS_LABELS[result.class_name] || result.class_name}
              </div>

              {result.stair_message && (
                <div style={{
                  background: "#1a0505", border: "2px solid #EF444444",
                  borderRadius: 12, padding: 16, marginBottom: 16,
                  fontSize: 18, color: "#EF4444", fontWeight: 600,
                }}>
                  {result.stair_message}
                </div>
              )}

              {agentResult && (
                <div style={{
                  background: "#141824", borderRadius: 12, padding: 16,
                  fontSize: 15, color: "#CBD5E1", lineHeight: 1.6,
                }}>
                  {agentResult.navigation_instruction}
                </div>
              )}
            </div>
          ) : (
            <div style={{
              width: 140, height: 140, borderRadius: "50%",
              border: "4px solid #00D4FF",
              animation: "pulse 0.6s ease-in-out infinite",
              display: "flex", alignItems: "center", justifyContent: "center",
            }}>
              <span style={{ color: "#00D4FF", fontSize: 14 }}>SCANNING</span>
            </div>
          )}
        </>
      )}

      {scanCount > 0 && (
        <div style={{ position: "fixed", bottom: 8, right: 12 }}>
          <span style={{ fontSize: 10, color: "#333" }}>{scanCount}</span>
        </div>
      )}
      {debug && (
        <div style={{ position: "fixed", bottom: 24, left: 0, right: 0, textAlign: "center" }}>
          <span style={{ fontSize: 11, color: "#64748B" }}>{debug}</span>
        </div>
      )}

      <style>{`@keyframes pulse{0%,100%{opacity:1;transform:scale(1)}50%{opacity:.4;transform:scale(1.08)}}`}</style>
    </div>
  );
}
