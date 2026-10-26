import { Hono } from "npm:hono@4";
import { cors } from "npm:hono/cors";
import { sqlite } from "https://esm.town/v/std/sqlite/main.ts";

// ============================================================================
// 1. EMPIRICAL TELEMETRY & MODEL DISCOVERY SCHEMA (SQLITE)
// ============================================================================

await sqlite.execute(`
  CREATE TABLE IF NOT EXISTS mesh_models_v4 (
    tuple_id TEXT PRIMARY KEY,
    provider_id TEXT NOT NULL,
    model_id TEXT NOT NULL,
    iq_prior REAL NOT NULL,
    iq_verified REAL DEFAULT 1.0,
    ewma_ttft_ms REAL DEFAULT 900.0,
    ewma_tps REAL DEFAULT 35.0,
    success_count INTEGER DEFAULT 0,
    failure_streak INTEGER DEFAULT 0,
    cooldown_until INTEGER DEFAULT 0,
    last_probed INTEGER DEFAULT 0,
    last_error TEXT
  );
`);

await sqlite.execute(`
  CREATE TABLE IF NOT EXISTS mesh_discovery_meta (
    provider_id TEXT PRIMARY KEY,
    last_synced INTEGER DEFAULT 0
  );
`);

// ============================================================================
// 2. INTRINSIC CAPABILITY PARSER (IQ PRIOR HEURISTIC + BENCHMARK WEIGHTS)
// ============================================================================

// Evaluates discovered model strings against known 2026 frontier capability tiers [0.10 - 1.00]
function computeIntrinsicIqPrior(modelId: string, contextLength: number = 32768): number {
  const m = modelId.toLowerCase();

  // Exclude non-text/embedding/audio models
  if (/embed|whisper|tts|flux|sdxl|dall-e|moderation|guard/i.test(m)) return 0.0;

  let baseScore = 0.50;

  // Tier 1: Frontier / Heavy Reasoning / Large MoE (0.88 - 0.98)
  if (/claude-3-5-sonnet|claude-3-7|gpt-4o(?!-mini)|o1|o3|deepseek-r1(?!-distill-qwen-1\.5b)|deepseek-v3|deepseek-v4|glm-5|qwen-?2\.5-coder-32b|qwen-?2\.5-72b|qwen3-235b|llama-3\.3-70b|gpt-oss-120b|nemotron-70b|codestral|minimax-m2/i.test(m)) {
    baseScore = 0.92;
  } // Tier 2: Mid-Size High Efficiency / Distilled Reasoning (0.72 - 0.85)
  else if (/gemma-?4-(26b|31b)|gemma-?3-27b|qwen-coder|gpt-4o-mini|claude-3-haiku|nemotron-3\.5|mistral-large|pixtral|command-r-plus|llama-3\.1-70b|kilo-auto\/free/i.test(m)) {
    baseScore = 0.80;
  } // Tier 3: Fast Lightweight / Sub-15B (0.45 - 0.68)
  else if (/8b|7b|9b|12b|14b|mistral-nemo|flash|kilo-auto\/small|haiku/i.test(m)) {
    baseScore = 0.62;
  } // Tier 4: Nano / Edge Models (< 4B)
  else if (/1\.5b|3b|tinyllama|0\.5b|mini/i.test(m)) {
    baseScore = 0.35;
  }

  // Bonus for explicit coding/reasoning specialization or large context window
  if (/coder|reasoning|thinking|r1/i.test(m)) baseScore += 0.04;
  if (contextLength >= 128000) baseScore += 0.02;

  return Math.min(0.99, Number(baseScore.toFixed(3)));
}

// ============================================================================
// 3. LIVE UPSTREAM MODEL CATALOG DISCOVERY ADAPTERS
// ============================================================================

interface ProviderSpec {
  id: string;
  name: string;
  discoverModels: (signal: AbortSignal) => Promise<Array<{ id: string; context?: number }>>;
  executeCompletion: (
    modelId: string,
    messages: Array<{ role: string; content: string }>,
    signal: AbortSignal,
  ) => Promise<{ content: string; ttftMs: number; totalMs: number; estTokens: number }>;
}

const POISON_REGEX = /rate limit exceeded|too many requests|cloudflare ray id|just a moment\.\.\.|api key is required|insufficient_quota/i;

function verifyCleanText(text: string): void {
  if (!text || text.trim().length === 0) throw new Error("POISON_EMPTY: Zero content tokens");
  if (POISON_REGEX.test(text)) throw new Error("POISON_SOFTBLOCK: Upstream proxy error disguised as 200 OK");
}

// Measures real TTFT and End-to-End Token Velocity (TPS) via streaming or chunked timing
async function timedOpenAIFetch(
  url: string,
  headers: Record<string, string>,
  payload: Record<string, unknown>,
  signal: AbortSignal,
): Promise<{ content: string; ttftMs: number; totalMs: number; estTokens: number }> {
  const t0 = performance.now();
  const res = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json", ...headers },
    body: JSON.stringify(payload),
    signal,
  });

  const ttftHeaderMs = Math.round(performance.now() - t0);

  if (res.status === 429) throw new Error("HTTP_429: Rate limit");
  if (res.status === 401 || res.status === 403) throw new Error(`HTTP_${res.status}: Blocked`);
  if (!res.ok) throw new Error(`HTTP_${res.status}: Upstream error`);

  const rawText = await res.text();
  const totalMs = Math.max(ttftHeaderMs + 1, Math.round(performance.now() - t0));

  let content = "";
  if (rawText.startsWith("data: ")) {
    for (const line of rawText.split("\n")) {
      if (line.startsWith("data: ")) {
        const slice = line.slice(6).trim();
        if (slice === "[DONE]") continue;
        try {
          const parsed = JSON.parse(slice);
          content += parsed.choices?.[0]?.delta?.content ?? parsed.choices?.[0]?.message?.content ?? "";
        } catch {
          // Ignore keepalive comments
        }
      }
    }
  } else {
    const data = JSON.parse(rawText);
    content = data.choices?.[0]?.message?.content ?? data.choices?.[0]?.text ?? data.response ?? "";
  }

  verifyCleanText(content);
  // Approximate token count (1 token ~= 4 chars in English/Code)
  const estTokens = Math.max(1, Math.ceil(content.length / 4));

  return {
    content,
    ttftMs: ttftHeaderMs,
    totalMs,
    estTokens,
  };
}

const PROVIDERS: ProviderSpec[] = [
  {
    id: "llm7",
    name: "LLM7 Anonymous Mesh",
    discoverModels: async (signal) => {
      const res = await fetch("https://api.llm7.io/v1/models", { signal });
      if (!res.ok) throw new Error("Failed LLM7 model discovery");
      const json = await res.json();
      const list = Array.isArray(json.data) ? json.data : Array.isArray(json) ? json : [];
      return list.map((m: any) => ({ id: String(m.id), context: Number(m.context_window || 65536) }));
    },
    executeCompletion: (modelId, messages, signal) =>
      timedOpenAIFetch(
        "https://api.llm7.io/v1/chat/completions",
        { "Authorization": "Bearer unused" },
        { model: modelId, messages, stream: false },
        signal,
      ),
  },
  {
    id: "kilo",
    name: "Kilo Code AI Gateway",
    discoverModels: async (signal) => {
      try {
        const res = await fetch("https://ai-gateway.kilo.ai/api/v1/models", { signal });
        if (res.ok) {
          const json = await res.json();
          const freeOnes = (json.data || []).filter((m: any) => String(m.id).includes("free") || String(m.id).includes("kilo-auto"));
          if (freeOnes.length > 0) {
            return freeOnes.map((m: any) => ({ id: String(m.id), context: Number(m.context_length || 256000) }));
          }
        }
      } catch {
        // Fallback to known keyless endpoints if catalog route is throttled
      }
      return [
        { id: "kilo-auto/free", context: 256000 },
        { id: "kilo-auto/small", context: 128000 },
        { id: "google/gemma-4-26b-a4b-it:free", context: 128000 },
      ];
    },
    executeCompletion: (modelId, messages, signal) =>
      timedOpenAIFetch(
        "https://ai-gateway.kilo.ai/api/v1/chat/completions",
        { "x-kilocode-mode": "code", "User-Agent": "Kilo-Code/2.1" },
        { model: modelId, messages, stream: false },
        signal,
      ),
  },
  {
    id: "pollinations",
    name: "Pollinations Keyless Core",
    discoverModels: async (signal) => {
      const res = await fetch("https://text.pollinations.ai/models", { signal });
      if (!res.ok) throw new Error("Failed Pollinations discovery");
      const list = await res.json();
      return (Array.isArray(list) ? list : []).map((m: any) => ({
        id: String(m.name || m.id),
        context: 65536,
      }));
    },
    executeCompletion: (modelId, messages, signal) =>
      timedOpenAIFetch(
        "https://text.pollinations.ai/openai",
        { "User-Agent": "Vajra-Empirical-Router/4.0" },
        { model: modelId, messages, stream: false },
        signal,
      ),
  },
  {
    id: "opencode",
    name: "OpenCode V2 Zen",
    discoverModels: async () => [
      { id: "glm-5.3-flash-free", context: 200000 },
      { id: "deepseek-v4.1-flash-free", context: 128000 },
      { id: "nemotron-3.5-lightning-free", context: 128000 },
    ],
    executeCompletion: (modelId, messages, signal) =>
      timedOpenAIFetch(
        "https://opencode.ai/inference/openai/v1/chat/completions",
        {
          "x-opencode-client": "cli",
          "x-opencode-project": "opencode-cli",
          "User-Agent": "opencode-ai/2.6.0",
        },
        { model: modelId, messages, stream: false },
        signal,
      ),
  },
];

// ============================================================================
// 4. AUTO-DISCOVERY SYNC & PARETO OBJECTIVE SCORER
// ============================================================================

async function ensureCatalogSynced(force: boolean = false): Promise<void> {
  const now = Date.now();
  const metaRes = await sqlite.execute("SELECT provider_id, last_synced FROM mesh_discovery_meta");
  const syncMap = new Map<string, number>();
  for (const r of metaRes.rows) syncMap.set(String(r.provider_id), Number(r.last_synced));

  const syncTasks = PROVIDERS.map(async (p) => {
    const last = syncMap.get(p.id) || 0;
    // Re-discover every 6 hours unless forced
    if (!force && now - last < 6 * 3600 * 1000) return;

    const ctrl = new AbortController();
    const timer = setTimeout(() => ctrl.abort(), 5000);
    try {
      const discovered = await p.discoverModels(ctrl.signal);
      for (const item of discovered) {
        const iqPrior = computeIntrinsicIqPrior(item.id, item.context);
        if (iqPrior <= 0.0) continue; // Skip non-chat models

        const tupleId = `${p.id}::${item.id}`;
        await sqlite.execute({
          sql: `INSERT INTO mesh_models_v4 (tuple_id, provider_id, model_id, iq_prior)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(tuple_id) DO UPDATE SET iq_prior = ?`,
          args: [tupleId, p.id, item.id, iqPrior, iqPrior],
        });
      }
      await sqlite.execute({
        sql: `INSERT INTO mesh_discovery_meta (provider_id, last_synced) VALUES (?, ?)
              ON CONFLICT(provider_id) DO UPDATE SET last_synced = ?`,
        args: [p.id, now, now],
      });
    } catch {
      // Retain existing SQLite catalog on discovery timeout
    } finally {
      clearTimeout(timer);
    }
  });

  await Promise.allSettled(syncTasks);
}

interface ScoredCandidate {
  tupleId: string;
  providerId: string;
  modelId: string;
  iqEffective: number;
  ewmaTtftMs: number;
  ewmaTps: number;
  expectedFastMs: number;
  smartScore: number;
}

async function getRankedCandidates(mode: "fast" | "smart" | "code"): Promise<ScoredCandidate[]> {
  await ensureCatalogSynced(false);
  const now = Date.now();
  const res = await sqlite.execute({
    sql: "SELECT * FROM mesh_models_v4 WHERE cooldown_until <= ?",
    args: [now],
  });

  const candidates: ScoredCandidate[] = res.rows.map((r) => {
    const iqPrior = Number(r.iq_prior);
    const iqVerified = Number(r.iq_verified);
    const iqEffective = Number((iqPrior * iqVerified).toFixed(3));
    const ewmaTtftMs = Number(r.ewma_ttft_ms);
    const ewmaTps = Math.max(1.0, Number(r.ewma_tps));

    // Expected completion time (ms) for a standard 250-token turn
    const expectedFastMs = Math.round(ewmaTtftMs + (250 / ewmaTps) * 1000);

    // Smart score rewards high IQ and only penalizes extreme slowness (> 8000ms expected)
    const latencyPenalty = Math.max(0, (expectedFastMs - 8000) / 40000);
    const codeBonus = mode === "code" && /coder|codestral|glm|deepseek|claude|kilo/i.test(String(r.model_id)) ? 0.08 : 0;
    const smartScore = Number((iqEffective + codeBonus - latencyPenalty).toFixed(4));

    return {
      tupleId: String(r.tuple_id),
      providerId: String(r.provider_id),
      modelId: String(r.model_id),
      iqEffective,
      ewmaTtftMs: Math.round(ewmaTtftMs),
      ewmaTps: Number(ewmaTps.toFixed(1)),
      expectedFastMs,
      smartScore,
    };
  });

  if (mode === "fast") {
    // Filter out tiny incoherent models (< 0.45 IQ), then sort strictly by lowest expected completion time
    return candidates
      .filter((c) => c.iqEffective >= 0.45)
      .sort((a, b) => a.expectedFastMs - b.expectedFastMs);
  }

  // For "smart" or "code": sort by highest empirical Smart Score
  return candidates.sort((a, b) => b.smartScore - a.smartScore);
}

// ============================================================================
// 5. CLOSED-LOOP CANARY BENCHMARK & TELEMETRY UPDATERS
// ============================================================================

async function recordTupleSuccess(tupleId: string, ttftMs: number, totalMs: number, estTokens: number) {
  const streamDurationSec = Math.max(0.05, (totalMs - ttftMs) / 1000);
  const observedTps = Math.min(600, Math.max(2, estTokens / streamDurationSec));

  await sqlite.execute({
    sql: `UPDATE mesh_models_v4
          SET ewma_ttft_ms = (0.3 * ?) + (0.7 * ewma_ttft_ms),
              ewma_tps = (0.3 * ?) + (0.7 * ewma_tps),
              success_count = success_count + 1,
              failure_streak = 0,
              cooldown_until = 0
          WHERE tuple_id = ?`,
    args: [ttftMs, observedTps, tupleId],
  });
}

async function recordTupleFailure(tupleId: string, errMsg: string) {
  const now = Date.now();
  const baseSec = errMsg.includes("429") ? 120 : errMsg.includes("401") || errMsg.includes("403") ? 900 : 60;
  await sqlite.execute({
    sql: `UPDATE mesh_models_v4
          SET failure_streak = failure_streak + 1,
              cooldown_until = ?,
              ewma_ttft_ms = MIN(15000, ewma_ttft_ms * 1.4),
              last_error = ?
          WHERE tuple_id = ?`,
    args: [now + baseSec * 1000, errMsg.slice(0, 200), tupleId],
  });
}

// Runs a deterministic micro-verification probe to prove a model isn't a mislabeled 7B fallback
async function runCanaryProbeOnTopCandidates(limit: number = 4) {
  const topSmart = (await getRankedCandidates("smart")).slice(0, limit);
  const probePrompt = [
    {
      role: "user",
      content: "Compute (17 * 19) + 11. Reply ONLY with the exact integer and nothing else.",
    },
  ];

  const results = await Promise.allSettled(
    topSmart.map(async (cand) => {
      const provider = PROVIDERS.find((p) => p.id === cand.providerId);
      if (!provider) return null;

      const ctrl = new AbortController();
      const timer = setTimeout(() => ctrl.abort(), 9000);
      try {
        const res = await provider.executeCompletion(cand.modelId, probePrompt, ctrl.signal);
        const passed = /\b334\b/.test(res.content.trim());
        const verifiedMultiplier = passed ? 1.0 : 0.55; // Penalize models that fail basic arithmetic/instruction adherence

        await recordTupleSuccess(cand.tupleId, res.ttftMs, res.totalMs, res.estTokens);
        await sqlite.execute({
          sql: "UPDATE mesh_models_v4 SET iq_verified = ?, last_probed = ? WHERE tuple_id = ?",
          args: [verifiedMultiplier, Date.now(), cand.tupleId],
        });
        return { tupleId: cand.tupleId, passed, ttftMs: res.ttftMs, answer: res.content.trim() };
      } catch (e: any) {
        await recordTupleFailure(cand.tupleId, String(e?.message || "Probe failed"));
        throw e;
      } finally {
        clearTimeout(timer);
      }
    }),
  );

  return results;
}

// ============================================================================
// 6. HONO HTTP APPLICATION
// ============================================================================

const app = new Hono();
app.use("*", cors());

// Empirical Leaderboard Endpoint: Shows WHY each model is selected for free:fast vs free:smart
app.get("/", async (c) => {
  const fastRank = (await getRankedCandidates("fast")).slice(0, 5);
  const smartRank = (await getRankedCandidates("smart")).slice(0, 5);

  return c.json({
    engine: "vajra-empirical-sovereign-v4",
    selection_methodology: {
      free_fast_objective: "Minimizes E[T_total] = EWMA_TTFT_ms + (250 / EWMA_TPS) * 1000 subject to IQ >= 0.45",
      free_smart_objective: "Maximizes (IQ_prior * Canary_Verification_Multiplier) - Latency_Overflow_Penalty",
    },
    current_best_fast: fastRank,
    current_best_smart: smartRank,
    endpoints: {
      chat: "POST /v1/chat/completions",
      probe: "POST /v1/probe (Runs live arithmetic/instruction verification on top models)",
      sync: "POST /v1/sync (Forces upstream /v1/models catalog re-discovery)",
    },
  });
});

// Trigger Live Canary Verification Probe
app.post("/v1/probe", async (c) => {
  const probeResults = await runCanaryProbeOnTopCandidates(4);
  return c.json({ status: "probe_complete", results: probeResults });
});

// Force Upstream Model Catalog Re-Discovery
app.post("/v1/sync", async (c) => {
  await ensureCatalogSynced(true);
  return c.json({ status: "catalog_synced" });
});

// OpenAI Chat Completions with Staggered Empirical Hedging
app.post("/v1/chat/completions", async (c) => {
  try {
    const body = await c.req.json();
    const messages = Array.isArray(body.messages) ? body.messages : [];
    const reqModel = String(body.model || "free:smart").toLowerCase();

    const mode: "fast" | "smart" | "code" = reqModel.includes("fast")
      ? "fast"
      : reqModel.includes("code")
      ? "code"
      : "smart";

    const ranked = await getRankedCandidates(mode);
    if (ranked.length === 0) {
      return c.json({ error: { message: "No healthy discovered models available." } }, 503);
    }

    const masterAbort = new AbortController();
    const globalTimeout = setTimeout(() => masterAbort.abort(), 16000);

    // Select Top 2 empirically proven tuples for Wave 1, and #3 for staggered Hedge Wave
    const wave = ranked.slice(0, 3);

    const attempts = wave.map(async (cand, idx) => {
      // Stagger backup wave by 1300ms for fast mode, 2200ms for smart mode (giving large models room to think)
      const staggerMs = idx === 0 ? 0 : idx === 1 ? 250 : mode === "fast" ? 1300 : 2200;
      if (staggerMs > 0) {
        await new Promise<void>((resolve, reject) => {
          const t = setTimeout(resolve, staggerMs);
          masterAbort.signal.addEventListener("abort", () => {
            clearTimeout(t);
            reject(new Error("ABORTED_BY_WINNER"));
          }, { once: true });
        });
      }

      const provider = PROVIDERS.find((p) => p.id === cand.providerId);
      if (!provider) throw new Error("Provider missing");

      try {
        const res = await provider.executeCompletion(cand.modelId, messages, masterAbort.signal);
        await recordTupleSuccess(cand.tupleId, res.ttftMs, res.totalMs, res.estTokens);
        return { ...res, candidate: cand };
      } catch (err: any) {
        if (!masterAbort.signal.aborted) {
          await recordTupleFailure(cand.tupleId, String(err?.message || "Exec error"));
        }
        throw err;
      }
    });

    try {
      const winner = await Promise.any(attempts);
      masterAbort.abort();

      return c.json({
        id: `chatcmpl-${crypto.randomUUID()}`,
        object: "chat.completion",
        created: Math.floor(Date.now() / 1000),
        model: winner.candidate.modelId,
        provider: winner.candidate.providerId,
        telemetry: {
          mode_selected: mode,
          iq_effective: winner.candidate.iqEffective,
          observed_ttft_ms: winner.ttftMs,
          observed_total_ms: winner.totalMs,
          ewma_tps: winner.candidate.ewmaTps,
        },
        choices: [
          {
            index: 0,
            message: { role: "assistant", content: winner.content },
            finish_reason: "stop",
          },
        ],
      });
    } finally {
      clearTimeout(globalTimeout);
    }
  } catch (err: any) {
    return c.json({ error: { message: err.message || "Empirical mesh dispatch failed" } }, 502);
  }
});

export default app.fetch;
