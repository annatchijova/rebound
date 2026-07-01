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

export default function Live({ backendUrl }) {
  const [status, setStatus] = useState("idle"); // idle | init | ready | scanning | result | error
  const [error, setError] = useState("");
  const [result, setResult] = useState(null);
  const [scanCount, setScanCount] = useState(0);
  const [agentResult, setAgentResult] = useState(null);

  const engineRef = useRef(null);
  const gyroRef = useRef(null);

  // Initialize sonar engine
  const initialize = useCallback(async () => {
    setStatus("init");
    setError("");
    try {
      const engine = new SonarEngine();
      await engine.init(backendUrl);
      engineRef.current = engine;

      const gyro = new GyroReader();
      await gyro.start();
      gyroRef.current = gyro;

      setStatus("ready");
    } catch (e) {
      setError(e.message);
      setStatus("error");
    }
  }, [backendUrl]);

  // Cleanup
  useEffect(() => {
    return () => {
      engineRef.current?.destroy();
      gyroRef.current?.stop();
    };
  }, []);

  // Scan: emit chirp, capture, predict, vibrate
  const scan = useCallback(async () => {
    if (!engineRef.current) return;
    setStatus("scanning");
    setError("");

    try {
      // 1. Capture audio (chirp + echo)
      const { audio, sampleRate } = await engineRef.current.capture();

      // 2. Send to /predict
      const pitch = gyroRef.current?.pitch || 0;
      const pred = await engineRef.current.predict(backendUrl, audio, sampleRate, pitch);
      setResult(pred);
      setScanCount((c) => c + 1);

      // 3. Vibrate
      vibrate(pred.stairs_detection?.is_stair ? "stair_alert" : "single_pulse");

      // 4. Send to /process for memory agent
      try {
        const procResp = await fetch(`${backendUrl}/process`, {
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
          setAgentResult(await procResp.json());
        }
      } catch {
        // Memory agent is optional — don't block on failure
      }

      setStatus("result");
    } catch (e) {
      setError(e.message);
      setStatus("error");
    }
  }, [backendUrl]);

  const color = result ? CLASS_COLORS[result.class_name] || "#00D4FF" : "#00D4FF";

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

      {/* Status area */}
      {status === "idle" && (
        <button onClick={initialize} style={btnStyle("#00D4FF")}>
          Enable microphone
        </button>
      )}

      {status === "init" && (
        <div style={{ color: "#64748B", fontSize: 14 }}>Initializing...</div>
      )}

      {status === "error" && (
        <div style={{ textAlign: "center" }}>
          <div style={{ color: "#EF4444", fontSize: 13, marginBottom: 12 }}>{error}</div>
          <button onClick={initialize} style={btnStyle("#F59E0B")}>Retry</button>
        </div>
      )}

      {/* Main scan button */}
      {(status === "ready" || status === "result") && (
        <>
          <button
            onClick={scan}
            style={{
              width: 160, height: 160, borderRadius: "50%",
              background: "none", border: `3px solid ${color}`,
              color, fontSize: 18, fontWeight: 700, cursor: "pointer",
              display: "flex", alignItems: "center", justifyContent: "center",
              boxShadow: `0 0 30px ${color}33`,
              transition: "all 0.3s",
              marginBottom: 24,
            }}
          >
            SCAN
          </button>

          {/* Scan counter */}
          <div style={{ fontSize: 11, color: "#64748B", marginBottom: 20 }}>
            Scans: {scanCount}
          </div>
        </>
      )}

      {status === "scanning" && (
        <div style={{
          width: 160, height: 160, borderRadius: "50%",
          border: "3px solid #00D4FF", display: "flex",
          alignItems: "center", justifyContent: "center",
          animation: "pulse 0.8s ease-in-out infinite",
          marginBottom: 24,
        }}>
          <span style={{ color: "#00D4FF", fontSize: 14 }}>Listening...</span>
          <style>{`@keyframes pulse { 0%,100%{opacity:1;transform:scale(1)} 50%{opacity:0.5;transform:scale(1.05)} }`}</style>
        </div>
      )}

      {/* Result display */}
      {result && (status === "result" || status === "ready") && (
        <div style={{
          background: "#141824", borderRadius: 12, padding: 20, width: "100%",
          maxWidth: 360, border: `1px solid ${color}33`,
        }}>
          {/* Class + distance */}
          <div style={{ textAlign: "center", marginBottom: 16 }}>
            <div style={{ fontSize: 28, fontWeight: 700, color }}>
              {result.distance_m.toFixed(1)} m
            </div>
            <div style={{ fontSize: 16, color, fontWeight: 600, marginTop: 4 }}>
              {CLASS_LABELS[result.class_name] || result.class_name}
            </div>
            <div style={{ fontSize: 11, color: "#64748B", marginTop: 4 }}>
              Confidence: {(result.confidence * 100).toFixed(0)}% | SNR: {result.snr_db} dB
            </div>
          </div>

          {/* Stair message */}
          {result.stair_message && (
            <div style={{
              background: "#1a0505", border: "1px solid #EF444433",
              borderRadius: 8, padding: 10, marginBottom: 12,
              fontSize: 13, color: "#EF4444",
            }}>
              {result.stair_message}
            </div>
          )}

          {/* Agent instruction */}
          {agentResult && (
            <div style={{
              background: "#0A0E1A", borderRadius: 8, padding: 10,
              fontSize: 13, color: "#CBD5E1", lineHeight: 1.5,
              marginBottom: 12,
            }}>
              {agentResult.navigation_instruction}
            </div>
          )}

          {/* Probabilities */}
          <div style={{ fontSize: 11, color: "#64748B", marginBottom: 6 }}>Probabilities</div>
          {Object.entries(result.probabilities).map(([cls, prob]) => (
            <div key={cls} style={{ display: "flex", alignItems: "center", gap: 6, marginBottom: 3 }}>
              <span style={{ fontSize: 10, color: "#64748B", width: 70 }}>
                {CLASS_LABELS[cls] || cls}
              </span>
              <div style={{ flex: 1, background: "#1E2A3A", borderRadius: 3, height: 6, overflow: "hidden" }}>
                <div style={{
                  width: `${prob * 100}%`, height: "100%",
                  background: CLASS_COLORS[cls] || "#64748B",
                  borderRadius: 3, transition: "width 0.3s",
                }} />
              </div>
              <span style={{ fontSize: 10, color: "#94A3B8", width: 30, textAlign: "right" }}>
                {(prob * 100).toFixed(0)}%
              </span>
            </div>
          ))}
        </div>
      )}

      {/* Backend URL (small, bottom) */}
      <div style={{ marginTop: "auto", paddingTop: 24, textAlign: "center" }}>
        <div style={{ fontSize: 10, color: "#475569" }}>
          Backend: {backendUrl}
        </div>
      </div>
    </div>
  );
}

function btnStyle(color) {
  return {
    background: color, color: "#0A0E1A", border: "none",
    borderRadius: 8, padding: "12px 28px", fontSize: 14,
    fontWeight: 700, cursor: "pointer", fontFamily: "monospace",
  };
}
