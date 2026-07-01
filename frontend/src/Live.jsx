import { useState, useRef, useEffect, useCallback } from "react";
import { SonarEngine, vibrate, GyroReader } from "./sonar.js";

const CLASS_COLORS = {
  open_space: "#00D4FF",
  nearby_wall: "#F59E0B",
  doorway: "#10B981",
  corner: "#F97316",
  corridor: "#A78BFA",
  stairs: "#EF4444",
};

const CLASS_LABELS = {
  open_space: "Open space",
  nearby_wall: "Nearby wall",
  doorway: "Doorway",
  corner: "Corner",
  corridor: "Corridor",
  stairs: "Stairs",
};

// Haptic patterns mapped to danger level
const HAPTIC_FOR_CLASS = {
  open_space: "none",
  nearby_wall: "continuous_high",
  doorway: "double_pulse_slow",
  corner: "continuous_low",
  corridor: "single_pulse",
  stairs: "stair_alert",
};

/**
 * Speak text aloud using Web Speech API.
 * Works on both Android Chrome and iOS Safari.
 */
function speak(text) {
  if (!window.speechSynthesis) return;
  window.speechSynthesis.cancel();
  var utterance = new SpeechSynthesisUtterance(text);
  utterance.rate = 1.1;
  utterance.lang = "en-US";
  window.speechSynthesis.speak(utterance);
}

/**
 * Build a short spoken instruction from prediction result.
 */
function buildSpokenInstruction(pred, agentInstruction) {
  if (agentInstruction) return agentInstruction;

  var label = CLASS_LABELS[pred.class_name] || pred.class_name;
  var dist = pred.distance_m.toFixed(1);

  if (pred.stair_message) return pred.stair_message;
  if (pred.class_name === "open_space") return "Clear path ahead.";
  if (pred.class_name === "nearby_wall") return "Wall at " + dist + " meters.";
  if (pred.class_name === "doorway") return "Doorway at " + dist + " meters.";
  if (pred.class_name === "corner") return "Corner at " + dist + " meters.";
  if (pred.class_name === "corridor") return "Corridor. Walls at " + dist + " meters.";
  return label + " at " + dist + " meters.";
}

const SCAN_INTERVAL_MS = 2000; // Scan every 2 seconds in auto mode

export default function Live({ backendUrl }) {
  const [status, setStatus] = useState("idle");
  const [error, setError] = useState("");
  const [result, setResult] = useState(null);
  const [scanCount, setScanCount] = useState(0);
  const [agentResult, setAgentResult] = useState(null);
  const [running, setRunning] = useState(false);

  const engineRef = useRef(null);
  const gyroRef = useRef(null);
  const loopRef = useRef(false);

  // Initialize sonar engine + start auto loop
  const startNavigation = useCallback(async () => {
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
      setStatus("running");
      setRunning(true);
      loopRef.current = true;
    } catch (e) {
      setError(e.message);
      setStatus("error");
    }
  }, [backendUrl]);

  // Stop navigation
  const stopNavigation = useCallback(() => {
    loopRef.current = false;
    setRunning(false);
    setStatus("stopped");
    speak("Navigation stopped.");
  }, []);

  // Single scan cycle
  const doScan = useCallback(async () => {
    if (!engineRef.current || !loopRef.current) return;

    try {
      var { audio, sampleRate } = await engineRef.current.capture();
      var pitch = gyroRef.current ? gyroRef.current.pitch : 0;
      var pred = await engineRef.current.predict(backendUrl, audio, sampleRate, pitch);

      setResult(pred);
      setScanCount(function (c) { return c + 1; });

      // Vibrate based on detected class
      vibrate(HAPTIC_FOR_CLASS[pred.class_name] || "single_pulse");

      // Get memory agent instruction
      var agentText = "";
      try {
        var procResp = await fetch(backendUrl + "/process", {
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
        if (procResp.ok) {
          var agentData = await procResp.json();
          setAgentResult(agentData);
          agentText = agentData.navigation_instruction;
        }
      } catch (_e) {
        // Memory agent optional
      }

      // Speak the result
      var spoken = buildSpokenInstruction(pred, agentText);
      speak(spoken);

    } catch (e) {
      // Don't stop loop on transient errors
      console.error("Scan error:", e);
    }
  }, [backendUrl]);

  // Auto-scan loop
  useEffect(() => {
    if (!running) return;

    var intervalId;

    // First scan immediately
    doScan().then(function () {
      // Then scan every SCAN_INTERVAL_MS
      intervalId = setInterval(function () {
        if (loopRef.current) doScan();
      }, SCAN_INTERVAL_MS);
    });

    return function () {
      if (intervalId) clearInterval(intervalId);
    };
  }, [running, doScan]);

  // Cleanup
  useEffect(function () {
    return function () {
      loopRef.current = false;
      if (engineRef.current) engineRef.current.destroy();
      if (gyroRef.current) gyroRef.current.stop();
    };
  }, []);

  var color = result ? CLASS_COLORS[result.class_name] || "#00D4FF" : "#00D4FF";

  return (
    <div style={{
      background: "#0A0E1A", color: "#E2E8F0", minHeight: "100vh",
      fontFamily: "monospace", display: "flex", flexDirection: "column",
      alignItems: "center", padding: 16, boxSizing: "border-box",
    }}>
      {/* Header */}
      <div style={{ textAlign: "center", marginBottom: 24 }}>
        <div style={{ fontSize: 24, fontWeight: 700, color: "#00D4FF", letterSpacing: 3 }}>
          REBOUND
        </div>
        <div style={{ fontSize: 11, color: "#64748B", marginTop: 4 }}>
          Biomimetic Sonar Navigation
        </div>
      </div>

      {/* Start button — large, accessible */}
      {status === "idle" && (
        <button onClick={startNavigation} style={{
          width: 200, height: 200, borderRadius: "50%",
          background: "#00D4FF", color: "#0A0E1A", border: "none",
          fontSize: 20, fontWeight: 700, cursor: "pointer",
          fontFamily: "monospace",
          boxShadow: "0 0 40px #00D4FF55",
        }}
        aria-label="Start navigation"
        >
          START
        </button>
      )}

      {status === "init" && (
        <div style={{ color: "#64748B", fontSize: 16 }}>Initializing...</div>
      )}

      {status === "error" && (
        <div style={{ textAlign: "center" }}>
          <div style={{ color: "#EF4444", fontSize: 13, marginBottom: 12, padding: 16 }}>{error}</div>
          <button onClick={startNavigation} style={{
            background: "#F59E0B", color: "#0A0E1A", border: "none",
            borderRadius: 8, padding: "12px 28px", fontSize: 14,
            fontWeight: 700, cursor: "pointer",
          }}>Retry</button>
        </div>
      )}

      {/* Running state — big stop button + live result */}
      {(status === "running" || status === "stopped") && (
        <>
          {/* Stop/Resume button — big and accessible */}
          <button
            onClick={running ? stopNavigation : startNavigation}
            style={{
              width: 160, height: 160, borderRadius: "50%",
              background: running ? "none" : "#10B981",
              border: running ? "3px solid " + color : "none",
              color: running ? color : "#0A0E1A",
              fontSize: 16, fontWeight: 700, cursor: "pointer",
              fontFamily: "monospace",
              boxShadow: running ? "0 0 30px " + color + "33" : "0 0 30px #10B98155",
              animation: running ? "pulse 1.5s ease-in-out infinite" : "none",
              marginBottom: 16,
            }}
            aria-label={running ? "Stop navigation" : "Resume navigation"}
          >
            {running ? "SCANNING" : "RESUME"}
          </button>

          <div style={{ fontSize: 11, color: "#64748B", marginBottom: 16 }}>
            Scans: {scanCount} {running ? "| Auto mode" : "| Stopped"}
          </div>

          <style>{`@keyframes pulse { 0%,100%{opacity:1;transform:scale(1)} 50%{opacity:0.7;transform:scale(1.03)} }`}</style>

          {/* Result display */}
          {result && (
            <div style={{
              background: "#141824", borderRadius: 12, padding: 20, width: "100%",
              maxWidth: 360, border: "1px solid " + color + "33",
            }}>
              {/* Class + distance — large for visibility */}
              <div style={{ textAlign: "center", marginBottom: 16 }}>
                <div style={{ fontSize: 48, fontWeight: 700, color: color }}>
                  {result.distance_m.toFixed(1)} m
                </div>
                <div style={{ fontSize: 20, color: color, fontWeight: 600, marginTop: 4 }}>
                  {CLASS_LABELS[result.class_name] || result.class_name}
                </div>
                <div style={{ fontSize: 11, color: "#64748B", marginTop: 4 }}>
                  {(result.confidence * 100).toFixed(0)}% confidence | SNR {result.snr_db} dB
                </div>
              </div>

              {/* Stair alert */}
              {result.stair_message && (
                <div style={{
                  background: "#1a0505", border: "1px solid #EF444433",
                  borderRadius: 8, padding: 12, marginBottom: 12,
                  fontSize: 15, color: "#EF4444", fontWeight: 600,
                }}>
                  {result.stair_message}
                </div>
              )}

              {/* Agent instruction */}
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
          Backend: {backendUrl}
        </div>
      </div>
    </div>
  );
}
