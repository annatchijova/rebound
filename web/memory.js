// Deterministic memory core for the REBOUND demo.
//
// Faithful client-side port of the Python Memory Agent's decision path
// (src/memory/{profile,episodic,semantic}.py). No LLM touches these values:
// they are pure arithmetic, reproducible run to run, exactly as on the backend.
// Qwen only narrates the result — it never changes it. That is why the demo can
// show the REAL memory learning with no server and no API key.

// --- UserProfile.update_implicit (src/memory/profile.py) ---
// An "advance" confirms the prediction, a "retreat" contradicts it. After the
// predicted class is adjusted, the weights are renormalized so their mean is 1.0
// (a Bayesian prior over the classifier softmax).
const IMPLICIT_FACTORS = {
  advance: 1.05, // confirms prediction
  hesitate: 0.98, // mild uncertainty
  retreat: 0.9, // prediction likely incorrect
  ignore: 1.0, // neutral
};

export function updateImplicit(weights, classId, action) {
  const factor = IMPLICIT_FACTORS[action] ?? 1.0;
  const next = weights.slice();
  next[classId] *= factor;
  const mean = next.reduce((a, b) => a + b, 0) / next.length;
  if (mean > 0) {
    for (let i = 0; i < next.length; i++) next[i] /= mean;
  }
  return next;
}

// --- EpisodicMemory (src/memory/episodic.py) ---
// Every stored episode starts at relevance 1.0. On each new store, existing
// episodes decay by decayRate; those that fall below relevanceThreshold are
// forgotten (selective forgetting); the buffer is capped at maxEpisodes.
export class EpisodicMemory {
  constructor({ maxEpisodes = 200, decayRate = 0.995, relevanceThreshold = 0.1 } = {}) {
    this.maxEpisodes = maxEpisodes;
    this.decayRate = decayRate;
    this.relevanceThreshold = relevanceThreshold;
    this.episodes = [];
  }

  store(episode) {
    for (const ep of this.episodes) ep.relevance *= this.decayRate;
    this.episodes.push({ ...episode, relevance: 1.0 });
    this.episodes = this.episodes.filter((ep) => ep.relevance >= this.relevanceThreshold);
    if (this.episodes.length > this.maxEpisodes) {
      this.episodes = this.episodes.slice(-this.maxEpisodes);
    }
  }

  get length() {
    return this.episodes.length;
  }
}

// --- SemanticMemory.consolidate_from_episodic (src/memory/semantic.py) ---
// "Consolidation during sleep": scan the episodic buffer and, for any class with
// at least minObservations events, record the behavioural patterns whose rate
// crosses a threshold (frequent hesitation, frequent retreat, confident
// advancing). Recomputed from the buffer each step — deterministic and idempotent.
const CLASS_LABELS = {
  open_space: "open space",
  nearby_wall: "nearby wall",
  doorway: "doorway",
  corner: "corner",
  corridor: "corridor",
};

export function consolidateSemantic(episodes, minObservations = 5) {
  const entries = {};
  if (episodes.length < minObservations) return entries;

  const classActions = {};
  for (const ep of episodes) {
    const cls = ep.predictionClass;
    classActions[cls] ??= {};
    classActions[cls][ep.userAction] = (classActions[cls][ep.userAction] ?? 0) + 1;
  }

  for (const [cls, actions] of Object.entries(classActions)) {
    const total = Object.values(actions).reduce((a, b) => a + b, 0);
    if (total < minObservations) continue;
    const label = CLASS_LABELS[cls] ?? cls;

    const hesitateRate = (actions.hesitate ?? 0) / total;
    if (hesitateRate > 0.3) {
      entries[`difficulty_${cls}`] = {
        value: `Hesitates frequently in ${label}`,
        confidence: Math.min(hesitateRate, 0.95),
        rate: hesitateRate,
      };
    }

    const retreatRate = (actions.retreat ?? 0) / total;
    if (retreatRate > 0.2) {
      entries[`retreat_pattern_${cls}`] = {
        value: `Retreats in ${label}`,
        confidence: Math.min(retreatRate, 0.95),
        rate: retreatRate,
      };
    }

    const advanceRate = (actions.advance ?? 0) / total;
    if (advanceRate > 0.7) {
      entries[`confident_${cls}`] = {
        value: `Advances confidently in ${label}`,
        confidence: Math.min(advanceRate, 0.95),
        rate: advanceRate,
      };
    }
  }

  return entries;
}
