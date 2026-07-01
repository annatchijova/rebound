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

function speak(text) {
  try {
    var s = window["speechSynthesis"];
    if (!s) return;
    s.cancel();
    var u = new (window["SpeechSynthesisUtterance"])(text);
    u.rate = 1.1;
    u.lang = "en-US";
    s.speak(u);
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
  var [phase, setPhase] = useState("start"); // start | ready | scanning | result | error
  var [error, setError] = useState("");
  var [result, setResult] = useState(null);
  var [scanCount, setScanCount] = useState(0);
  var [agentResult, setAgentResult] = useState(null);

  var engineRef = useRef(null);
  var gyroRef = useRef(null);
  var scanningRef = useRef(false);
  var wakeLockRef = useRef(null);

  // Initialize on first tap
  async function handleStart() {
    setPhase("scanning");
    setError("");
    try {
      var engine = new SonarEngine();
      await engine.init(backendUrl);
      engineRef.current = engine;

      var gyro = new GyroReader();
      await gyro.start();
      gyroRef.current = gyro;

      // Keep screen on
      try {
        if (navigator.wakeLock) {
          wakeLockRef.current = await navigator.wakeLock.request("screen");
        }
      } catch (_) {}

      speak("REBOUND ready. Tap anywhere to scan.");
      setPhase("ready");
    } catch (e) {
      setError(e.message);
      setPhase("error");
    }
  }

  // Scan on tap — the entire screen triggers this
  async function handleScan() {
    if (phase !== "ready" && phase !== "result") return;
    if (scanningRef.current) return; // prevent double-tap overlap
    scanningRef.current = true;
    setPhase("scanning");

    try {
      var cap = await engineRef.current.capture();
      var pitch = gyroRef.current ? gyroRef.current.pitch : 0;
      var pred = await engineRef.current.predict(backendUrl, cap.audio, cap.sampleRate, pitch);

      setResult(pred);
      setScanCount(function(c) { return c + 1; });

      // Vibrate
      vibrate(HAPTIC_FOR_CLASS[pred.class_name] || "single_pulse");

      // Memory agent
      var agentText = "";
      try {
        var resp = await fetch(backendUrl + "/process", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            user_id: "mobile_user",
            prediction: { class_name: pred.class_name, confidence: pred.confidence, distance_m: pred.distance_m },
            features_summary: pred.features_summary,
            user_action: "advance",
            session_id: 1,
          }),
        });
        if (resp.ok) {
          var data = await resp.json();
          setAgentResult(data);
          agentText = data.navigation_instruction;
        }
      } catch (_) {}

      // Speak
      speak(buildSpoken(pred, agentText));
      setPhase("result");
    } catch (e) {
      speak("Error. Try again.");
      setError(e.message);
      setPhase("error");
    }

    scanningRef.current = false;
  }

  // Cleanup
  useEffect(function() {
    return function() {
      if (engineRef.current) engineRef.current.destroy();
      if (gyroRef.current) gyroRef.current.stop();
      if (wakeLockRef.current) wakeLockRef.current.release();
    };
  }, []);

  var color = result ? (CLASS_COLORS[result.class_name] || "#00D4FF") : "#00D4FF";

  // START screen — full screen button
  if (phase === "start") {
    return (
      <div onClick={handleStart} style={{
        background: "#0A0E1A", color: "#E2E8F0", minHeight: "100vh",
        display: "flex", flexDirection: "column", alignItems: "center",
        justifyContent: "center", fontFamily: "monospace", cursor: "pointer",
        userSelect: "none", WebkitUserSelect: "none",
      }}>
        <div style={{ fontSize: 32, fontWeight: 700, color: "#00D4FF", letterSpacing: 3, marginBottom: 16 }}>
          REBOUND
        </div>
        <div style={{
          width: 180, height: 180, borderRadius: "50%", background: "#00D4FF",
          display: "flex", alignItems: "center", justifyContent: "center",
          boxShadow: "0 0 60px #00D4FF44", marginBottom: 24,
        }}>
          <span style={{ color: "#0A0E1A", fontSize: 22, fontWeight: 700 }}>TAP TO START</span>
        </div>
        <div style={{ fontSize: 12, color: "#64748B", textAlign: "center", padding: "0 32px" }}>
          After starting, tap anywhere on screen to scan your surroundings.
          Results will be spoken aloud.
        </div>
      </div>
    );
  }

  // ERROR screen
  if (phase === "error") {
    return (
      <div onClick={handleStart} style={{
        background: "#0A0E1A", color: "#E2E8F0", minHeight: "100vh",
        display: "flex", flexDirection: "column", alignItems: "center",
        justifyContent: "center", fontFamily: "monospace", cursor: "pointer",
        userSelect: "none", WebkitUserSelect: "none",
      }}>
        <div style={{ color: "#EF4444", fontSize: 14, marginBottom: 16, padding: 24, textAlign: "center", wordBreak: "break-word" }}>
          {error}
        </div>
        <div style={{
          background: "#F59E0B", color: "#0A0E1A", borderRadius: 12,
          padding: "16px 32px", fontSize: 16, fontWeight: 700,
        }}>TAP TO RETRY</div>
      </div>
    );
  }

  // MAIN screen — entire screen is tap target
  return (
    <div
      onClick={handleScan}
      style={{
        background: "#0A0E1A", color: "#E2E8F0", minHeight: "100vh",
        display: "flex", flexDirection: "column", alignItems: "center",
        justifyContent: "center", fontFamily: "monospace", cursor: "pointer",
        userSelect: "none", WebkitUserSelect: "none", padding: 16,
      }}
    >
      {/* Scanning indicator */}
      {phase === "scanning" && (
        <div style={{
          width: 140, height: 140, borderRadius: "50%",
          border: "3px solid #00D4FF",
          animation: "pulse 0.6s ease-in-out infinite",
          display: "flex", alignItems: "center", justifyContent: "center",
          marginBottom: 24,
        }}>
          <span style={{ color: "#00D4FF", fontSize: 14 }}>SCANNING</span>
        </div>
      )}

      {/* Result display */}
      {phase === "result" && result && (
        <>
          {/* Distance — huge */}
          <div style={{ fontSize: 72, fontWeight: 700, color: color, lineHeight: 1 }}>
            {result.distance_m.toFixed(1)}
          </div>
          <div style={{ fontSize: 16, color: "#64748B", marginBottom: 8 }}>meters</div>

          {/* Class */}
          <div style={{ fontSize: 28, fontWeight: 700, color: color, marginBottom: 8 }}>
            {CLASS_LABELS[result.class_name] || result.class_name}
          </div>

          {/* Confidence */}
          <div style={{ fontSize: 12, color: "#64748B", marginBottom: 16 }}>
            {(result.confidence * 100).toFixed(0)}% confidence
          </div>

          {/* Stair alert */}
          {result.stair_message && (
            <div style={{
              background: "#1a0505", border: "1px solid #EF444444",
              borderRadius: 12, padding: 16, marginBottom: 16,
              fontSize: 16, color: "#EF4444", fontWeight: 600,
              width: "100%", maxWidth: 340, textAlign: "center",
            }}>
              {result.stair_message}
            </div>
          )}

          {/* Agent instruction */}
          {agentResult && (
            <div style={{
              background: "#141824", borderRadius: 12, padding: 16,
              fontSize: 14, color: "#CBD5E1", lineHeight: 1.6,
              width: "100%", maxWidth: 340, textAlign: "center",
              marginBottom: 16,
            }}>
              {agentResult.navigation_instruction}
            </div>
          )}

          {/* Tap hint */}
          <div style={{ fontSize: 13, color: "#475569", marginTop: 8 }}>
            Tap anywhere to scan again
          </div>
        </>
      )}

      {/* Scan counter + ready state */}
      {phase === "ready" && (
        <div style={{ textAlign: "center" }}>
          <div style={{
            width: 140, height: 140, borderRadius: "50%",
            border: "3px solid #00D4FF33",
            display: "flex", alignItems: "center", justifyContent: "center",
            marginBottom: 24,
          }}>
            <span style={{ color: "#00D4FF", fontSize: 16 }}>TAP TO SCAN</span>
          </div>
        </div>
      )}

      <div style={{ position: "fixed", bottom: 8, left: 0, right: 0, textAlign: "center" }}>
        <span style={{ fontSize: 10, color: "#333" }}>Scans: {scanCount}</span>
      </div>

      <style>{`@keyframes pulse{0%,100%{opacity:1;transform:scale(1)}50%{opacity:.5;transform:scale(1.05)}}`}</style>
    </div>
  );
}
