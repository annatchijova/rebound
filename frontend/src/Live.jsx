import { useState, useRef, useEffect, useCallback } from "react";
import { SonarEngine, vibrate, GyroReader } from "./sonar.js";

var CLASS_COLORS = {
  open_space: "#00D4FF",
  nearby_wall: "#F59E0B",
  doorway: "#10B981",
  corner: "#F97316",
  corridor: "#A78BFA",
  stairs: "#EF4444",
};

var CLASS_LABELS = {
  open_space: "Open space",
  nearby_wall: "Nearby wall",
  doorway: "Doorway",
  corner: "Corner",
  corridor: "Corridor",
  stairs: "Stairs",
};

var HAPTIC_FOR_CLASS = {
  open_space: "none",
  nearby_wall: "continuous_high",
  doorway: "double_pulse_slow",
  corner: "continuous_low",
  corridor: "single_pulse",
  stairs: "stair_alert",
};

function speak(text) {
  try {
    var synth = window["speechSynthesis"];
    if (!synth) return;
    synth.cancel();
    var u = new (window["SpeechSynthesisUtterance"])(text);
    u.rate = 1.0;
    u.lang = "en-US";
    synth.speak(u);
  } catch (_e) {
    // TTS not available
  }
}

function buildSpoken(pred, agentText) {
  if (agentText) return agentText;
  if (pred.stair_message) return pred.stair_message;
  var d = pred.distance_m.toFixed(1);
  var map = {
    open_space: "Clear path ahead.",
    nearby_wall: "Wall at " + d + " meters.",
    doorway: "Doorway at " + d + " meters.",
    corner: "Corner at " + d + " meters.",
    corridor: "Corridor. Walls at " + d + " meters.",
    stairs: "Stairs detected.",
  };
  return map[pred.class_name] || pred.class_name + " at " + d + " meters.";
}

var SCAN_DELAY = 2500;

export default function Live({ backendUrl }) {
  var [status, setStatus] = useState("idle");
  var [error, setError] = useState("");
  var [result, setResult] = useState(null);
  var [scanCount, setScanCount] = useState(0);
  var [agentResult, setAgentResult] = useState(null);

  var engineRef = useRef(null);
  var gyroRef = useRef(null);
  var runningRef = useRef(false);
  var timerRef = useRef(null);

  // One scan cycle
  async function doOneScan() {
    var engine = engineRef.current;
    if (!engine || !runningRef.current) return;

    try {
      var capture = await engine.capture();
      if (!runningRef.current) return;

      var pitch = gyroRef.current ? gyroRef.current.pitch : 0;
      var pred = await engine.predict(backendUrl, capture.audio, capture.sampleRate, pitch);
      if (!runningRef.current) return;

      setResult(pred);
      setScanCount(function (c) { return c + 1; });

      // Vibrate
      vibrate(HAPTIC_FOR_CLASS[pred.class_name] || "single_pulse");

      // Memory agent (non-blocking)
      var agentText = "";
      try {
        var resp = await fetch(backendUrl + "/process", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            user_id: "mobile_user",
            prediction: {
              class_name: pred.class_name,
              confidence: pred.confidence,
              distance_m: pred.distance_m,
            },
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

      // Speak result
      speak(buildSpoken(pred, agentText));

    } catch (e) {
      console.error("Scan error:", e);
    }

    // Schedule next scan
    if (runningRef.current) {
      timerRef.current = setTimeout(doOneScan, SCAN_DELAY);
    }
  }

  // Start
  var startNavigation = useCallback(async function () {
    setStatus("init");
    setError("");
    try {
      var engine = new SonarEngine();
      await engine.init(backendUrl);
      engineRef.current = engine;

      var gyro = new GyroReader();
      await gyro.start();
      gyroRef.current = gyro;

      speak("REBOUND activated. Scanning.");
      runningRef.current = true;
      setStatus("running");

      // Start scan loop
      timerRef.current = setTimeout(doOneScan, 500);
    } catch (e) {
      setError(e.message);
      setStatus("error");
    }
  }, [backendUrl]);

  // Stop
  var stopNavigation = useCallback(function () {
    runningRef.current = false;
    if (timerRef.current) clearTimeout(timerRef.current);
    setStatus("stopped");
    speak("Stopped.");
  }, []);

  // Resume
  var resumeNavigation = useCallback(function () {
    runningRef.current = true;
    setStatus("running");
    speak("Resuming.");
    timerRef.current = setTimeout(doOneScan, 500);
  }, []);

  // Cleanup
  useEffect(function () {
    return function () {
      runningRef.current = false;
      if (timerRef.current) clearTimeout(timerRef.current);
      if (engineRef.current) engineRef.current.destroy();
      if (gyroRef.current) gyroRef.current.stop();
    };
  }, []);

  var color = result ? CLASS_COLORS[result.class_name] || "#00D4FF" : "#00D4FF";
  var isRunning = status === "running";

  return (
    <div style={{
      background: "#0A0E1A", color: "#E2E8F0", minHeight: "100vh",
      fontFamily: "monospace", display: "flex", flexDirection: "column",
      alignItems: "center", padding: 16, boxSizing: "border-box",
    }}>
      <div style={{ textAlign: "center", marginBottom: 24 }}>
        <div style={{ fontSize: 24, fontWeight: 700, color: "#00D4FF", letterSpacing: 3 }}>
          REBOUND
        </div>
        <div style={{ fontSize: 11, color: "#64748B", marginTop: 4 }}>
          Biomimetic Sonar Navigation
        </div>
      </div>

      {status === "idle" && (
        <button onClick={startNavigation} style={{
          width: 200, height: 200, borderRadius: "50%",
          background: "#00D4FF", color: "#0A0E1A", border: "none",
          fontSize: 20, fontWeight: 700, cursor: "pointer",
          fontFamily: "monospace", boxShadow: "0 0 40px #00D4FF55",
        }} aria-label="Start navigation">
          START
        </button>
      )}

      {status === "init" && (
        <div style={{ color: "#64748B", fontSize: 16 }}>Initializing...</div>
      )}

      {status === "error" && (
        <div style={{ textAlign: "center" }}>
          <div style={{ color: "#EF4444", fontSize: 13, marginBottom: 12, padding: 16,
            wordBreak: "break-word" }}>{error}</div>
          <button onClick={startNavigation} style={{
            background: "#F59E0B", color: "#0A0E1A", border: "none",
            borderRadius: 8, padding: "12px 28px", fontSize: 14,
            fontWeight: 700, cursor: "pointer",
          }}>Retry</button>
        </div>
      )}

      {(status === "running" || status === "stopped") && (
        <>
          <button
            onClick={isRunning ? stopNavigation : resumeNavigation}
            style={{
              width: 160, height: 160, borderRadius: "50%",
              background: isRunning ? "none" : "#10B981",
              border: isRunning ? "3px solid " + color : "none",
              color: isRunning ? color : "#0A0E1A",
              fontSize: 16, fontWeight: 700, cursor: "pointer",
              fontFamily: "monospace",
              boxShadow: isRunning ? "0 0 30px " + color + "33" : "0 0 30px #10B98155",
              animation: isRunning ? "pulse 1.5s ease-in-out infinite" : "none",
              marginBottom: 16,
            }}
            aria-label={isRunning ? "Stop" : "Resume"}
          >
            {isRunning ? "SCANNING" : "RESUME"}
          </button>

          <div style={{ fontSize: 11, color: "#64748B", marginBottom: 16 }}>
            Scans: {scanCount}
          </div>

          <style>{`@keyframes pulse { 0%,100%{opacity:1;transform:scale(1)} 50%{opacity:0.7;transform:scale(1.03)} }`}</style>

          {result && (
            <div style={{
              background: "#141824", borderRadius: 12, padding: 20, width: "100%",
              maxWidth: 360, border: "1px solid " + color + "33",
            }}>
              <div style={{ textAlign: "center", marginBottom: 16 }}>
                <div style={{ fontSize: 48, fontWeight: 700, color: color }}>
                  {result.distance_m.toFixed(1)} m
                </div>
                <div style={{ fontSize: 20, color: color, fontWeight: 600, marginTop: 4 }}>
                  {CLASS_LABELS[result.class_name] || result.class_name}
                </div>
                <div style={{ fontSize: 11, color: "#64748B", marginTop: 4 }}>
                  {(result.confidence * 100).toFixed(0)}% | SNR {result.snr_db} dB
                </div>
              </div>

              {result.stair_message && (
                <div style={{
                  background: "#1a0505", border: "1px solid #EF444433",
                  borderRadius: 8, padding: 12, marginBottom: 12,
                  fontSize: 15, color: "#EF4444", fontWeight: 600,
                }}>
                  {result.stair_message}
                </div>
              )}

              {agentResult && (
                <div style={{
                  background: "#0A0E1A", borderRadius: 8, padding: 12,
                  fontSize: 14, color: "#CBD5E1", lineHeight: 1.5,
                }}>
                  {agentResult.navigation_instruction}
                </div>
              )}
            </div>
          )}
        </>
      )}

      <div style={{ marginTop: "auto", paddingTop: 24, textAlign: "center" }}>
        <div style={{ fontSize: 10, color: "#475569" }}>
          {backendUrl}
        </div>
      </div>
    </div>
  );
}
