import { useState, useRef, useEffect } from "react";
import { SonarEngine, vibrate, GyroReader } from "./sonar.js";

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

// TTS — dos vías:
// speakNow: SOLO dentro del handler del gesto, sin awaits antes. Desbloquea el motor.
// speak: para después de awaits, con el motor ya desbloqueado por speakNow.
function speakNow(text) {
  try {
    var s = window["speechSynthesis"];
    if (!s) return;
    s.resume();
    var u = new (window["SpeechSynthesisUtterance"])(text);
    u.lang = "en-US";
    u.rate = 1.1;
    s.speak(u);
  } catch (_) {}
}

function speak(text) {
  try {
    var s = window["speechSynthesis"];
    if (!s) return;
    s.cancel();
    s.resume();
    var u = new (window["SpeechSynthesisUtterance"])(text);
    u.lang = "en-US";
    u.rate = 1.1;
    setTimeout(function () { s.speak(u); }, 80);
  } catch (_) {}
}

function buildSpoken(pred, agentText) {
  if (agentText) return agentText;
  if (pred.stair_message) return pred.stair_message;
  var d = pred.distance_m.toFixed(1);
  if (pred.class_name === "open_space") return "Clear path.";
  if (pred.class_name === "stairs") return "Stairs detected.";
  return (CLASS_LABELS[pred.class_name] || pred.class_name) + ". " + d + " meters.";
}

export default function Live({ backendUrl }) {
  var [phase, setPhase] = useState("start");
  var [error, setError] = useState("");
  var [result, setResult] = useState(null);
  var [scanCount, setScanCount] = useState(0);
  var [agentResult, setAgentResult] = useState(null);
  var [debug, setDebug] = useState("");

  var engineRef = useRef(null);
  var gyroRef = useRef(null);
  var scanningRef = useRef(false);
  var initRef = useRef(false);

  // INITIALIZE — first tap
  async function handleStart() {
    if (initRef.current) return;
    initRef.current = true;
    speakNow("Starting");
    setPhase("init");
    setError("");
    var engine = null;
    try {
      engine = new SonarEngine();
      await engine.init(backendUrl);
      engineRef.current = engine;

      var gyro = new GyroReader();
      await gyro.start();
      gyroRef.current = gyro;

      try { if (navigator.wakeLock) await navigator.wakeLock.request("screen"); } catch (_) {}

      speak("Ready. Tap to scan.");
      setPhase("ready");
    } catch (e) {
      if (engine) { try { engine.destroy(); } catch (_) {} }
      initRef.current = false;
      setError(String(e && e.message ? e.message : e));
      setPhase("error");
    }
  }

  // SCAN — every subsequent tap
  async function handleTap() {
    if (phase === "start") { handleStart(); return; }
    if (phase === "error") {
      if (!engineRef.current) { handleStart(); return; }
      setError("");
      setPhase("ready");
      return;
    }
    if (phase === "init") return;
    if (scanningRef.current) return;
    scanningRef.current = true;

    speakNow("Scanning");
    setPhase("scanning");

    try {
      var t0 = Date.now();
      var cap = await engineRef.current.capture();
      var pitch = gyroRef.current ? gyroRef.current.pitch : 0;
      var pred = await engineRef.current.predict(backendUrl, cap.audio, cap.sampleRate, pitch);
      var ms = Date.now() - t0;

      setResult(pred);
      setAgentResult(null);
      setScanCount(function (c) { return c + 1; });
      setPhase("result");
      vibrate(HAPTIC_FOR_CLASS[pred.class_name] || "single_pulse");
      speak(buildSpoken(pred, ""));
      setDebug(
        pred.class_name + " " + Math.round(pred.confidence * 100) + "% · SNR " +
        pred.snr_db + " dB · " + ms + " ms"
      );

      // Memory agent en segundo plano
      fetch(backendUrl + "/process", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          user_id: "mobile_user",
          prediction: { class_name: pred.class_name, confidence: pred.confidence, distance_m: pred.distance_m },
          features_summary: pred.features_summary,
          user_action: "advance", session_id: 1,
        }),
      })
        .then(function (r) { return r.ok ? r.json() : null; })
        .then(function (data) {
          if (data) { setAgentResult(data); speak(data.navigation_instruction); }
        })
        .catch(function (_) {});
    } catch (e) {
      setError(String(e && e.message ? e.message : e));
      setPhase("error");
      speak("Error");
    }
    scanningRef.current = false;
  }

  // Cleanup
  useEffect(function() {
    return function() {
      if (engineRef.current) engineRef.current.destroy();
      if (gyroRef.current) gyroRef.current.stop();
    };
  }, []);

  var color = result ? (CLASS_COLORS[result.class_name] || "#00D4FF") : "#00D4FF";

  return (
    <div
      onClick={handleTap}
      onTouchEnd={function(e) { e.preventDefault(); handleTap(); }}
      style={{
        background: "#0A0E1A", color: "#E2E8F0", minHeight: "100vh", width: "100%",
        display: "flex", flexDirection: "column", alignItems: "center",
        justifyContent: "center", fontFamily: "monospace",
        cursor: "pointer", userSelect: "none", WebkitUserSelect: "none",
        WebkitTapHighlightColor: "transparent", padding: 20,
      }}
    >
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
              TAP ANYWHERE TO START
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

      {phase === "scanning" && (
        <div style={{
          width: 160, height: 160, borderRadius: "50%",
          border: "4px solid #00D4FF",
          animation: "pulse 0.5s ease-in-out infinite",
          display: "flex", alignItems: "center", justifyContent: "center",
        }}>
          <span style={{ color: "#00D4FF", fontSize: 16, fontWeight: 600 }}>SCANNING</span>
        </div>
      )}

      {phase === "ready" && (
        <div style={{ textAlign: "center" }}>
          <div style={{
            width: 160, height: 160, borderRadius: "50%",
            border: "4px solid #00D4FF33",
            display: "flex", alignItems: "center", justifyContent: "center",
            marginBottom: 16,
          }}>
            <span style={{ color: "#00D4FF", fontSize: 16 }}>TAP TO SCAN</span>
          </div>
          <div style={{ fontSize: 12, color: "#64748B" }}>Tap anywhere on screen</div>
        </div>
      )}

      {phase === "result" && result && (
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
              marginBottom: 16,
            }}>
              {agentResult.navigation_instruction}
            </div>
          )}

          <div style={{ fontSize: 14, color: "#475569", marginTop: 16 }}>
            tap anywhere to scan again
          </div>
        </div>
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
