import { useState, useEffect, useRef, useCallback } from "react";
import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer } from "recharts";
import { updateImplicit, EpisodicMemory, consolidateSemantic } from "./memory.js";

const CLASSES = ["open_space", "nearby_wall", "doorway", "corner", "corridor"];
const CLASS_LABELS = ["Open space", "Nearby wall", "Doorway", "Corner", "Corridor"];
const CLASS_COLORS = ["#00D4FF", "#F59E0B", "#10B981", "#F97316", "#A78BFA"];

// A single navigation session. Interleaved so the sonar view varies, and long
// enough that each class crosses the 5-observation floor the real semantic
// consolidation requires. The behavioural story the memory agent should learn:
// confident in open space and corridors, hesitant at doorways, retreats at
// corners, and grows confident with walls over the session.
const SEQUENCE = [
  { class_id: 0, action: "advance",  dist: 4.2, rt60: 0.45, centroid: 7200, echo: -18 },
  { class_id: 4, action: "advance",  dist: 0.9, rt60: 0.21, centroid: 6850, echo: -8  },
  { class_id: 2, action: "hesitate", dist: 0.4, rt60: 0.11, centroid: 6300, echo: -3  },
  { class_id: 1, action: "hesitate", dist: 0.6, rt60: 0.13, centroid: 6550, echo: -4  },
  { class_id: 3, action: "retreat",  dist: 0.5, rt60: 0.10, centroid: 6150, echo: -3  },
  { class_id: 0, action: "advance",  dist: 4.0, rt60: 0.46, centroid: 7150, echo: -18 },
  { class_id: 2, action: "hesitate", dist: 0.4, rt60: 0.10, centroid: 6250, echo: -3  },
  { class_id: 4, action: "advance",  dist: 0.8, rt60: 0.22, centroid: 6800, echo: -8  },
  { class_id: 1, action: "advance",  dist: 0.7, rt60: 0.13, centroid: 6600, echo: -4  },
  { class_id: 3, action: "hesitate", dist: 0.6, rt60: 0.10, centroid: 6200, echo: -3  },
  { class_id: 2, action: "hesitate", dist: 0.4, rt60: 0.11, centroid: 6300, echo: -3  },
  { class_id: 0, action: "advance",  dist: 4.1, rt60: 0.44, centroid: 7100, echo: -17 },
  { class_id: 4, action: "advance",  dist: 0.9, rt60: 0.20, centroid: 6900, echo: -7  },
  { class_id: 1, action: "advance",  dist: 0.7, rt60: 0.14, centroid: 6500, echo: -5  },
  { class_id: 2, action: "advance",  dist: 0.4, rt60: 0.11, centroid: 6300, echo: -3  },
  { class_id: 3, action: "retreat",  dist: 0.5, rt60: 0.09, centroid: 6100, echo: -2  },
  { class_id: 0, action: "advance",  dist: 4.3, rt60: 0.45, centroid: 7200, echo: -18 },
  { class_id: 2, action: "hesitate", dist: 0.4, rt60: 0.10, centroid: 6250, echo: -3  },
  { class_id: 4, action: "advance",  dist: 0.8, rt60: 0.21, centroid: 6850, echo: -8  },
  { class_id: 1, action: "advance",  dist: 0.7, rt60: 0.13, centroid: 6600, echo: -4  },
  { class_id: 3, action: "hesitate", dist: 0.6, rt60: 0.10, centroid: 6200, echo: -3  },
  { class_id: 2, action: "hesitate", dist: 0.4, rt60: 0.11, centroid: 6300, echo: -3  },
  { class_id: 0, action: "advance",  dist: 4.0, rt60: 0.46, centroid: 7150, echo: -17 },
  { class_id: 4, action: "advance",  dist: 0.9, rt60: 0.20, centroid: 6900, echo: -8  },
  { class_id: 1, action: "advance",  dist: 0.7, rt60: 0.13, centroid: 6550, echo: -4  },
  { class_id: 3, action: "retreat",  dist: 0.5, rt60: 0.09, centroid: 6100, echo: -3  },
  { class_id: 2, action: "hesitate", dist: 0.4, rt60: 0.10, centroid: 6250, echo: -3  },
  { class_id: 0, action: "advance",  dist: 4.2, rt60: 0.45, centroid: 7200, echo: -18 },
];

// Snake-case class names, matching the semantic keys the memory core produces
// (difficulty_doorway, confident_open_space, ...).
const CLASS_NAME = ["open_space", "nearby_wall", "doorway", "corner", "corridor"];
const CLASS_PHRASE = ["Open space", "Wall nearby", "Doorway", "Corner", "Corridor"];

// Templated narration derived from the REAL memory state. This is the only place
// an LLM would slot in (Qwen phrases the same facts when the backend is up); it
// never changes the sealed numbers — priors, counts and patterns are already fixed.
function narrate({ classId, action, dist, newKeys, semantic }) {
  const where = CLASS_PHRASE[classId];
  const d = dist.toFixed(1);
  if (newKeys.length) {
    const e = semantic[newKeys[0]];
    return `${where} at ${d} m. Memory consolidated — ${e.value.toLowerCase()} (${Math.round(e.rate * 100)}%). Personalizing guidance.`;
  }
  const cls = CLASS_NAME[classId];
  const known = [`difficulty_${cls}`, `retreat_pattern_${cls}`, `confident_${cls}`]
    .map((k) => semantic[k])
    .find(Boolean);
  const base = {
    advance: `${where} at ${d} m. Path clear — advance.`,
    hesitate: `${where} at ${d} m. Approach with caution.`,
    retreat: `${where} at ${d} m. Reorient before proceeding.`,
    ignore: `${where} at ${d} m.`,
  }[action];
  return known ? `${base} (${known.value.toLowerCase()} — profile applied).` : base;
}

const HAPTIC_LABELS = {
  pulse_slow: "Slow pulse — safe",
  pulse_fast: "Fast pulse — caution",
  double_tap: "Double tap — obstacle",
  long_vibration: "Long vibration — stop",
  triple_tap: "Triple tap — reorient",
};

// Haptic pattern derived from the class and the user's action.
function hapticFor(classId, action) {
  if (action === "retreat") return "long_vibration"; // stop
  if (action === "hesitate") return classId === 3 ? "triple_tap" : "double_tap";
  if (classId === 2) return "double_tap"; // doorway — narrow frame
  return "pulse_slow"; // open space / corridor / cleared wall
}

function SonarViz({ classId, ping, distance }) {
  const color = CLASS_COLORS[classId] ?? "#00D4FF";
  const rings = [120, 90, 60, 30];

  return (
    <svg viewBox="0 0 240 240" style={{ width: "100%", maxWidth: 240 }}>
      {rings.map((r, i) => (
        <circle
          key={i}
          cx={120} cy={120} r={r}
          fill="none"
          stroke="#1E2A3A"
          strokeWidth={i === 0 ? 1.5 : 0.8}
        />
      ))}
      {ping && (
        <circle
          cx={120} cy={120} r={30}
          fill="none"
          stroke={color}
          strokeWidth={2}
          opacity={0.7}
          style={{ animation: "sonar-ping 1.2s ease-out forwards" }}
        />
      )}
      <circle cx={120} cy={120} r={8} fill={color} opacity={0.9} />
      <line x1={120} y1={112} x2={120} y2={20} stroke={color} strokeWidth={1.5} opacity={0.5} />
      <text x={120} y={210} textAnchor="middle" fill="#94A3B8" fontSize={11}>
        {distance > 0 ? `${distance.toFixed(1)} m` : "—"}
      </text>
      <text x={120} y={228} textAnchor="middle" fill={color} fontSize={12} fontWeight={500}>
        {CLASS_LABELS[classId]}
      </text>
      <style>{`
        @keyframes sonar-ping {
          0%   { r: 30; opacity: 0.7; }
          100% { r: 120; opacity: 0; }
        }
      `}</style>
    </svg>
  );
}

function ConfBar({ weights }) {
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
      {CLASSES.map((cls, i) => (
        <div key={cls} style={{ display: "flex", alignItems: "center", gap: 8 }}>
          <span style={{ fontSize: 10, color: "#64748B", width: 80, flexShrink: 0 }}>
            {CLASS_LABELS[i]}
          </span>
          <div style={{ flex: 1, background: "#1E2A3A", borderRadius: 4, height: 10, overflow: "hidden" }}>
            <div style={{
              width: `${Math.min(100, (weights[i] / 2) * 100)}%`,
              height: "100%",
              background: CLASS_COLORS[i],
              borderRadius: 4,
              transition: "width 0.5s ease",
              opacity: 0.85,
            }} />
          </div>
          <span style={{ fontSize: 10, color: CLASS_COLORS[i], width: 32, textAlign: "right" }}>
            {weights[i]?.toFixed(2)}
          </span>
        </div>
      ))}
    </div>
  );
}

function HapticIcon({ pattern }) {
  const color = pattern === "long_vibration" ? "#EF4444"
    : pattern === "double_tap" ? "#F59E0B"
    : "#10B981";
  return (
    <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
      <div style={{
        width: 10, height: 10, borderRadius: "50%",
        background: color,
        boxShadow: `0 0 8px ${color}`,
        animation: "blink 1s ease-in-out infinite",
      }} />
      <span style={{ fontSize: 12, color }}>{HAPTIC_LABELS[pattern] ?? pattern}</span>
      <style>{`@keyframes blink { 0%,100%{opacity:1} 50%{opacity:0.3} }`}</style>
    </div>
  );
}

export default function ReboundUI() {
  const [backendUrl, setBackendUrl] = useState("http://localhost:8000");
  const [connected, setConnected] = useState(null);
  const [step, setStep] = useState(-1);
  const [ping, setPing] = useState(false);
  const [profileHistory, setProfileHistory] = useState([]);
  const [weights, setWeights] = useState([1, 1, 1, 1, 1]);
  const [episodicCount, setEpisodicCount] = useState(0);
  const [semanticCount, setSemanticCount] = useState(0);
  const [instruction, setInstruction] = useState("Start demo to begin navigation session.");
  const [hapticPattern, setHapticPattern] = useState("pulse_slow");
  const [loading, setLoading] = useState(false);
  const [autoPlay, setAutoPlay] = useState(false);
  const [sessionId, setSessionId] = useState(1);
  const [memoryOps, setMemoryOps] = useState([]);
  const autoRef = useRef(null);
  // Real memory state that lives across steps (not just re-render state).
  const episodicRef = useRef(new EpisodicMemory());
  const semanticRef = useRef({});
  const [semanticEntries, setSemanticEntries] = useState({});

  const current = step >= 0 ? SEQUENCE[step] : null;

  const testConnection = async () => {
    try {
      const r = await fetch(`${backendUrl}/health`, { signal: AbortSignal.timeout(3000) });
      if (r.ok) { setConnected(true); } else { setConnected(false); }
    } catch { setConnected(false); }
  };

  const triggerPing = () => {
    setPing(false);
    setTimeout(() => setPing(true), 50);
    setTimeout(() => setPing(false), 1300);
  };

  // Run one step through the REAL deterministic memory core — no backend, no
  // LLM. Same math as src/memory/*.py; Qwen would only phrase the result.
  const applyLocalStep = (s) => {
    const obs = SEQUENCE[s];

    // 1) Adaptive Bayesian prior — real UserProfile.update_implicit
    const newWeights = updateImplicit(weights, obs.class_id, obs.action);
    setWeights(newWeights);

    // 2) Episodic memory — real store with temporal decay and forgetting
    const ep = episodicRef.current;
    ep.store({
      sessionId,
      predictionClass: CLASSES[obs.class_id],
      userAction: obs.action,
      distanceM: obs.dist,
      confidence: 0.75,
    });
    setEpisodicCount(ep.length);

    // 3) Semantic consolidation — real consolidate_from_episodic
    const prevKeys = new Set(Object.keys(semanticRef.current));
    const semantic = consolidateSemantic(ep.episodes);
    const newKeys = Object.keys(semantic).filter((k) => !prevKeys.has(k));
    semanticRef.current = semantic;
    setSemanticEntries(semantic);
    setSemanticCount(Object.keys(semantic).length);

    // 4) Narration, haptic and memory ops derived from the real state
    setInstruction(narrate({ classId: obs.class_id, action: obs.action, dist: obs.dist, newKeys, semantic }));
    setHapticPattern(hapticFor(obs.class_id, obs.action));
    const ops = [{ op: obs.action === "advance" ? "reinforce" : "flag", key: CLASS_LABELS[obs.class_id] }];
    for (const k of newKeys) ops.push({ op: "consolidate", key: k });
    setMemoryOps(ops);

    // 5) Profile evolution chart
    setProfileHistory((prev) => [...prev, {
      step: s + 1,
      open: +newWeights[0].toFixed(2),
      wall: +newWeights[1].toFixed(2),
      door: +newWeights[2].toFixed(2),
      corner: +newWeights[3].toFixed(2),
      corridor: +newWeights[4].toFixed(2),
    }]);
  };

  const applyRealStep = async (s) => {
    const obs = SEQUENCE[s];
    try {
      const r = await fetch(`${backendUrl}/process`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          user_id: "demo_user",
          prediction: {
            class_name: CLASSES[obs.class_id],
            confidence: 0.75 + Math.random() * 0.2,
            distance_m: obs.dist,
          },
          features_summary: {
            rt60: obs.rt60,
            spectral_centroid: obs.centroid,
            echo_strength: obs.echo,
          },
          user_action: obs.action,
          session_id: sessionId,
        }),
      });
      const data = await r.json();
      setInstruction(data.navigation_instruction);
      setHapticPattern(data.haptic_pattern);
      setMemoryOps(data.memory_ops?.slice(0, 3) ?? []);

      const prof = await fetch(`${backendUrl}/profile/demo_user`);
      const pd = await prof.json();
      const w = pd.profile?.class_weights ?? weights;
      setWeights(w);
      setEpisodicCount(pd.episodic_stats?.total ?? s + 1);
      setSemanticCount(pd.semantic?.length ?? Math.floor(s / 4));
      setProfileHistory(prev => [...prev, {
        step: s + 1,
        open: +w[0].toFixed(2),
        wall: +w[1].toFixed(2),
        door: +w[2].toFixed(2),
        corner: +w[3].toFixed(2),
        corridor: +w[4].toFixed(2),
      }]);
    } catch {
      applyLocalStep(s);
    }
  };

  const advance = useCallback(async () => {
    const next = step + 1;
    if (next >= SEQUENCE.length) { setAutoPlay(false); return; }
    setStep(next);
    setLoading(true);
    triggerPing();
    if (connected) { await applyRealStep(next); } else { applyLocalStep(next); }
    setLoading(false);
  }, [step, connected, weights]);

  const reset = () => {
    setStep(-1);
    setWeights([1, 1, 1, 1, 1]);
    setProfileHistory([]);
    setEpisodicCount(0);
    setSemanticCount(0);
    episodicRef.current = new EpisodicMemory();
    semanticRef.current = {};
    setSemanticEntries({});
    setInstruction("Start demo to begin navigation session.");
    setHapticPattern("pulse_slow");
    setMemoryOps([]);
    setAutoPlay(false);
    setSessionId(s => s + 1);
  };

  useEffect(() => {
    if (autoPlay) {
      autoRef.current = setTimeout(() => advance(), 1800);
    }
    return () => clearTimeout(autoRef.current);
  }, [autoPlay, step, advance]);

  const s = { background: "#0A0E1A", color: "#E2E8F0", minHeight: "100vh", fontFamily: "monospace", padding: 16, boxSizing: "border-box" };

  return (
    <div style={s}>
      {/* Header */}
      <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 16, flexWrap: "wrap", gap: 8 }}>
        <div>
          <span style={{ fontSize: 20, fontWeight: 700, color: "#00D4FF", letterSpacing: 2 }}>REBOUND</span>
          <span style={{ fontSize: 11, color: "#64748B", marginLeft: 12 }}>Biomimetic Sonar · Qwen MemoryAgent · Qwen Cloud Hackathon</span>
        </div>
        <div style={{ display: "flex", gap: 8, alignItems: "center", flexWrap: "wrap" }}>
          <input
            value={backendUrl}
            onChange={e => setBackendUrl(e.target.value)}
            style={{ background: "#141824", border: "1px solid #1E2A3A", color: "#E2E8F0", borderRadius: 6, padding: "4px 10px", fontSize: 12, width: 200 }}
            placeholder="http://localhost:8000"
          />
          <button onClick={testConnection} style={{ background: "#141824", border: "1px solid #1E2A3A", color: "#94A3B8", borderRadius: 6, padding: "4px 12px", fontSize: 12, cursor: "pointer" }}>
            Test connection
          </button>
          <span style={{ fontSize: 11, color: connected === true ? "#10B981" : "#00D4FF" }}>
            {connected === true ? "● Live backend · Qwen" : "● Local memory · deterministic"}
          </span>
        </div>
      </div>

      {/* Main grid */}
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: 12, marginBottom: 12 }}>

        {/* Left: Sonar */}
        <div style={{ background: "#141824", borderRadius: 10, padding: 16, border: "1px solid #1E2A3A" }}>
          <div style={{ fontSize: 11, color: "#64748B", marginBottom: 8, textTransform: "uppercase", letterSpacing: 1 }}>Sonar</div>
          <SonarViz classId={current?.class_id ?? 0} ping={ping} distance={current?.dist ?? 0} />
          <div style={{ marginTop: 12 }}>
            <div style={{ fontSize: 11, color: "#64748B", marginBottom: 4 }}>User action</div>
            <span style={{ fontSize: 13, color: current?.action === "advance" ? "#10B981" : current?.action === "retreat" ? "#EF4444" : "#F59E0B", fontWeight: 600 }}>
              {current?.action?.toUpperCase() ?? "—"}
            </span>
          </div>
          <div style={{ marginTop: 8 }}>
            <div style={{ fontSize: 11, color: "#64748B", marginBottom: 4 }}>RT60</div>
            <span style={{ fontSize: 13, color: "#E2E8F0" }}>{current ? `${current.rt60.toFixed(2)} s` : "—"}</span>
          </div>
        </div>

        {/* Center: Instruction */}
        <div style={{ background: "#141824", borderRadius: 10, padding: 16, border: "1px solid #1E2A3A", display: "flex", flexDirection: "column", gap: 12 }}>
          <div style={{ fontSize: 11, color: "#64748B", textTransform: "uppercase", letterSpacing: 1 }}>Navigation · Step {step >= 0 ? step + 1 : 0}/{SEQUENCE.length}</div>
          <div style={{ fontSize: 32, fontWeight: 700, color: "#00D4FF", lineHeight: 1 }}>
            {current ? `${current.dist.toFixed(1)} m` : "— m"}
          </div>
          <div style={{ flex: 1, fontSize: 13, color: "#CBD5E1", lineHeight: 1.6, background: "#0A0E1A", borderRadius: 8, padding: 12, minHeight: 80 }}>
            {loading ? <span style={{ color: "#64748B" }}>Processing...</span> : instruction}
          </div>
          <div>
            <div style={{ fontSize: 11, color: "#64748B", marginBottom: 6 }}>Haptic output</div>
            <HapticIcon pattern={hapticPattern} />
          </div>
          {memoryOps.length > 0 && (
            <div>
              <div style={{ fontSize: 11, color: "#64748B", marginBottom: 4 }}>Memory ops</div>
              {memoryOps.map((op, i) => (
                <div key={i} style={{ fontSize: 11, color: "#A78BFA" }}>
                  {op.op}: {op.key ?? op.value ?? ""}
                </div>
              ))}
            </div>
          )}
        </div>

        {/* Right: Memory state */}
        <div style={{ background: "#141824", borderRadius: 10, padding: 16, border: "1px solid #1E2A3A", display: "flex", flexDirection: "column", gap: 12 }}>
          <div style={{ fontSize: 11, color: "#64748B", textTransform: "uppercase", letterSpacing: 1 }}>Memory agent · Qwen-Max</div>
          <div style={{ display: "flex", gap: 16 }}>
            <div>
              <div style={{ fontSize: 10, color: "#64748B" }}>Episodic</div>
              <div style={{ fontSize: 22, color: "#00D4FF", fontWeight: 700 }}>{episodicCount}</div>
            </div>
            <div>
              <div style={{ fontSize: 10, color: "#64748B" }}>Semantic</div>
              <div style={{ fontSize: 22, color: "#A78BFA", fontWeight: 700 }}>{semanticCount}</div>
            </div>
          </div>
          <div>
            <div style={{ fontSize: 11, color: "#64748B", marginBottom: 8 }}>Bayesian priors</div>
            <ConfBar weights={weights} />
          </div>
          <div>
            <div style={{ fontSize: 11, color: "#64748B", marginBottom: 8 }}>Learned patterns · semantic memory</div>
            {Object.keys(semanticEntries).length === 0 ? (
              <div style={{ fontSize: 11, color: "#475569", fontStyle: "italic" }}>
                Consolidating — needs 5+ observations of a space.
              </div>
            ) : (
              <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
                {Object.entries(semanticEntries).map(([key, e]) => {
                  const c = key.startsWith("confident_") ? "#10B981"
                    : key.startsWith("retreat_pattern_") ? "#EF4444" : "#F59E0B";
                  return (
                    <div key={key} style={{ display: "flex", alignItems: "center", gap: 8, background: "#0A0E1A", borderRadius: 6, padding: "6px 8px" }}>
                      <span style={{ width: 6, height: 6, borderRadius: "50%", background: c, flexShrink: 0 }} />
                      <span style={{ fontSize: 11, color: "#CBD5E1", flex: 1 }}>{e.value}</span>
                      <span style={{ fontSize: 11, color: c, fontWeight: 700 }}>{Math.round(e.rate * 100)}%</span>
                    </div>
                  );
                })}
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Profile evolution chart */}
      <div style={{ background: "#141824", borderRadius: 10, padding: 16, border: "1px solid #1E2A3A", marginBottom: 12 }}>
        <div style={{ fontSize: 11, color: "#64748B", marginBottom: 12, textTransform: "uppercase", letterSpacing: 1 }}>User profile evolution — adaptive Bayesian priors</div>
        <ResponsiveContainer width="100%" height={120}>
          <LineChart data={profileHistory}>
            <XAxis dataKey="step" tick={{ fill: "#64748B", fontSize: 10 }} />
            <YAxis domain={[0, 2]} tick={{ fill: "#64748B", fontSize: 10 }} width={28} />
            <Tooltip contentStyle={{ background: "#141824", border: "1px solid #1E2A3A", borderRadius: 6, fontSize: 11 }} />
            <Line type="monotone" dataKey="open"    stroke={CLASS_COLORS[0]} dot={false} strokeWidth={1.5} name="Open space" />
            <Line type="monotone" dataKey="wall"    stroke={CLASS_COLORS[1]} dot={false} strokeWidth={1.5} name="Nearby wall" />
            <Line type="monotone" dataKey="door"    stroke={CLASS_COLORS[2]} dot={false} strokeWidth={1.5} name="Doorway" />
            <Line type="monotone" dataKey="corner"  stroke={CLASS_COLORS[3]} dot={false} strokeWidth={1.5} name="Corner" />
            <Line type="monotone" dataKey="corridor" stroke={CLASS_COLORS[4]} dot={false} strokeWidth={1.5} name="Corridor" />
          </LineChart>
        </ResponsiveContainer>
      </div>

      {/* Controls */}
      <div style={{ display: "flex", gap: 8, justifyContent: "center" }}>
        <button
          onClick={advance}
          disabled={loading || step >= SEQUENCE.length - 1}
          style={{ background: "#00D4FF", color: "#0A0E1A", border: "none", borderRadius: 8, padding: "10px 24px", fontSize: 13, fontWeight: 700, cursor: "pointer", opacity: loading ? 0.5 : 1 }}
        >
          {step < 0 ? "Start session" : "Next step →"}
        </button>
        <button
          onClick={() => setAutoPlay(a => !a)}
          disabled={step >= SEQUENCE.length - 1}
          style={{ background: autoPlay ? "#F59E0B" : "#141824", color: autoPlay ? "#0A0E1A" : "#94A3B8", border: "1px solid #1E2A3A", borderRadius: 8, padding: "10px 20px", fontSize: 13, cursor: "pointer" }}
        >
          {autoPlay ? "⏸ Pause" : "▶ Auto"}
        </button>
        <button
          onClick={reset}
          style={{ background: "#141824", color: "#94A3B8", border: "1px solid #1E2A3A", borderRadius: 8, padding: "10px 20px", fontSize: 13, cursor: "pointer" }}
        >
          Reset
        </button>
      </div>
    </div>
  );
}
