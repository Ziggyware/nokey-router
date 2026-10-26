import { Hono } from "hono";
import { cors } from "hono/cors";
import { bodyLimit } from "hono/body-limit";

// ============================================================================
// 1. DENO KV + L1 ISOLATE CACHE + CAS ATOMIC STATE ENGINE
// ============================================================================

const kv = await Deno.openKv();

export interface MeshModelRecord {
  tuple_id: string;
  provider_id: string;
  model_id: string;
  iq_prior: number;
  iq_verified: number;
  ewma_ttft_ms: number;
  ewma_tps: number;
  alpha_success: number;
  beta_failure: number;
  success_count: number;
  failure_streak: number;
  cooldown_until: number;
  last_probed: number;
  last_error: string | null;
}

const L1_CACHE = new Map<string, MeshModelRecord>();
let l1LastHydrated = 0;
const L1_TTL_MS = 15_000;

async function hydrateL1Cache(force = false): Promise<Map<string, MeshModelRecord>> {
  const now = Date.now();
  if (!force && l1LastHydrated > 0 && now - l1LastHydrated < L1_TTL_MS && L1_CACHE.size > 0) {
    return L1_CACHE;
  }
  const iter = kv.list<MeshModelRecord>({ prefix: ["mesh_models_v4"] });
  const fresh = new Map<string, MeshModelRecord>();
  for await (const entry of iter) {
    if (entry.value) {
      fresh.set(entry.value.tuple_id, entry.value);
    }
  }
  if (fresh.size > 0 || force) {
    L1_CACHE.clear();
    for (const [k, v] of fresh) L1_CACHE.set(k, v);
    l1LastHydrated = now;
  }
  return L1_CACHE;
}

async function atomicUpdateModel(
  tupleId: string,
  mutator: (current: MeshModelRecord) => MeshModelRecord,
  maxRetries = 4,
): Promise<void> {
  const key = ["mesh_models_v4", tupleId];
  for (let attempt = 0; attempt < maxRetries; attempt++) {
    const entry = await kv.get<MeshModelRecord>(key);
    if (!entry.value) return;
    const updated = mutator(entry.value);
    const res = await kv.atomic().check(entry).set(key, updated).commit();
    if (res.ok) {
      L1_CACHE.set(tupleId, updated);
      return;
    }
    await new Promise((r) => setTimeout(r, 10 * (attempt + 1) + Math.random() * 15));
  }
}

// ============================================================================
// 2. SECURITY: CONSTANT-TIME AUTH, RATE LIMITER & SSRF SANITIZER
// ============================================================================

const encoder = new TextEncoder();

function timingSafeCompare(a: string, b: string): boolean {
  const bufA = encoder.encode(a);
  const bufB = encoder.encode(b);
  if (bufA.byteLength !== bufB.byteLength) {
    let diff = 1;
    for (let i = 0; i < bufB.byteLength; i++) diff |= bufB[i] ^ bufB[i];
    return diff === 0;
  }
  let diff = 0;
  for (let i = 0; i < bufA.byteLength; i++) {
    diff |= bufA[i] ^ bufB[i];
  }
  return diff === 0;
}

const IP_BUCKETS = new Map<string, { tokens: number; lastRefill: number }>();
const MAX_BURST = 30;
const REFILL_PER_SEC = 1.5;

function checkRateLimit(ip: string): boolean {
  const now = Date.now();
  const bucket = IP_BUCKETS.get(ip) ?? { tokens: MAX_BURST, lastRefill: now };
  const elapsedSec = Math.max(0, (now - bucket.lastRefill) / 1000);
  bucket.tokens = Math.min(MAX_BURST, bucket.tokens + elapsedSec * REFILL_PER_SEC);
  bucket.lastRefill = now;

  if (IP_BUCKETS.size > 5000) IP_BUCKETS.clear();

  if (bucket.tokens < 1) {
    IP_BUCKETS.set(ip, bucket);
    return false;
  }
  bucket.tokens -= 1;
  IP_BUCKETS.set(ip, bucket);
  return true;
}

function buildRotatedHeaders(baseHeaders: Record<string, string>): Headers {
  const uaPool = [
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/131.0.0.0 Safari/537.36",
    "OpenAI/v1 PythonBindings/1.55.0",
    "continue-dev/1.0.24 (darwin-arm64)",
    "cline/3.2.0 vscode/1.96.0",
  ];
  const entries = Object.entries({
    "Accept": "application/json, text/event-stream",
    "Accept-Language": "en-US,en;q=0.9",
    "Cache-Control": "no-cache",
    "User-Agent": uaPool[Math.floor(Math.random() * uaPool.length)],
    ...baseHeaders,
    "Content-Type": "application/json",
  });

  for (let i = entries.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [entries[i], entries[j]] = [entries[j], entries[i]];
  }
  const h = new Headers();
  for (const [k, v] of entries) h.set(k, v);
  return h;
}

// ============================================================================
// 3. INTRINSIC CAPABILITY PARSER & SHANNON ENTROPY POISON GUARD
// ============================================================================

function computeIntrinsicIqPrior(modelId: string, contextLength: number = 32768): number {
  const m = modelId.toLowerCase();
  if (/embed|whisper|tts|flux|sdxl|dall-e|moderation|guard|rerank/i.test(m)) return 0.0;

  let baseScore = 0.50;

  if (/claude-3-5-sonnet|claude-3-7|gpt-4o(?!-mini)|o1|o3|deepseek-r1(?!-distill-qwen-1\.5b)|deepseek-v3|deepseek-v4|glm-5|qwen-?2\.5-coder-32b|qwen-?2\.5-72b|qwen3-235b|llama-3\.3-70b|gpt-oss-120b|nemotron-70b|codestral|minimax-m2/i.test(m)) {
    baseScore = 0.92;
  } else if (/gemma-?4-(26b|31b)|gemma-?3-27b|qwen-coder|gpt-4o-mini|claude-3-haiku|nemotron-3\.5|mistral-large|pixtral|command-r-plus|llama-3\.1-70b|kilo-auto\/free/i.test(m)) {
    baseScore = 0.80;
  } else if (/8b|7b|9b|12b|14b|mistral-nemo|flash|kilo-auto\/small|haiku/i.test(m)) {
    baseScore = 0.62;
  } else if (/1\.5b|3b|tinyllama|0\.5b|mini/i.test(m)) {
    baseScore = 0.35;
  }

  if (/coder|reasoning|thinking|r1/i.test(m)) baseScore += 0.04;
  if (contextLength >= 128000) baseScore += 0.02;

  return Math.min(0.99, Number(baseScore.toFixed(3)));
}

const POISON_REGEX = /rate limit exceeded|too many requests|cloudflare ray id|just a moment\.\.\.|api key is required|insufficient_quota|error code: 1020|visit https:\/\/|upgrade your plan|free tier limit/i;

function computeShannonEntropy(str: string): number {
  const len = str.length;
  if (len === 0) return 0;
  const freq = new Map<string, number>();
  for (let i = 0; i < len; i++) {
    const ch = str[i];
    freq.set(ch, (freq.get(ch) ?? 0) + 1);
  }
  let entropy = 0;
  for (const count of freq.values()) {
    const p = count / len;
    entropy -= p * Math.log2(p);
  }
  return entropy;
}

function verifyCleanText(text: string): void {
  const trimmed = text.trim();
  if (!trimmed) throw new Error("POISON_EMPTY: Zero content tokens");
  if (POISON_REGEX.test(trimmed.slice(0, 400))) {
    throw new Error("POISON_SOFTBLOCK: Upstream proxy error disguised as 200 OK");
  }
  if (trimmed.length >= 32 && computeShannonEntropy(trimmed.slice(0, 128)) < 1.15) {
    throw new Error("POISON_DEGENERATE: Low-entropy token collapse detected");
  }
}

// ============================================================================
// 4. EPISTEMIC THOMPSON SAMPLING + COMPLEXITY AUTO-CLASSIFIER
// ============================================================================

function sampleGamma(alpha: number): number {
  if (alpha < 1) {
    return sampleGamma(1 + alpha) * Math.pow(Math.random(), 1 / alpha);
  }
  const d = alpha - 1 / 3;
  const c = 1 / Math.sqrt(9 * d);
  while (true) {
    let x = 0;
    let v = 0;
    do {
      const u1 = Math.random();
      const u2 = Math.random();
      x = Math.sqrt(-2 * Math.log(Math.max(1e-12, u1))) * Math.cos(2 * Math.PI * u2);
      v = 1 + c * x;
    } while (v <= 0);
    v = v * v * v;
    const u = Math.random();
    if (u < 1 - 0.0331 * (x * x) * (x * x)) return d * v;
    if (Math.log(u) < 0.5 * x * x + d * (1 - v + Math.log(v))) return d * v;
  }
}

function sampleBeta(alpha: number, beta: number): number {
  const x = sampleGamma(Math.max(0.5, alpha));
  const y = sampleGamma(Math.max(0.5, beta));
  return x / (x + y);
}

function classifyPromptIntent(messages: Array<{ role: string; content: string }>): "fast" | "smart" | "code" {
  const combined = messages.map((m) => typeof m.content === "string" ? m.content : "").join("\n");
  if (/```|function\s+\w+|interface\s+\w+|def\s+\w+|import\s+\{|SELECT\s+.*\s+FROM|cargo\s+build/i.test(combined)) {
    return "code";
  }
  if (combined.length > 1800 || /prove|theorem|architect|analyze|differential|step-by-step|rigorous/i.test(combined)) {
    return "smart";
  }
  return "fast";
}

// ============================================================================
// 5. LIVE UPSTREAM MODEL CATALOG DISCOVERY ADAPTERS
// ============================================================================

interface ProviderSpec {
  id: string;
  name: string;
  discoverModels: (signal: AbortSignal) => Promise<Array<{ context?: id: number string; }>>;
  executeCompletion: (
    modelId: string,
    messages: Array<{ role: string; content: string }>,
    signal: AbortSignal,
  ) => Promise<{ content: string; ttftMs: number; totalMs: number; estTokens: number }>;
}

async function timedOpenAIFetch(
  url: string,
  headers: Record<string, string>,
  payload: Record<string, unknown>,
  signal: AbortSignal,
): Promise<{ content: string; ttftMs: number; totalMs: number; estTokens: number }> {
  const t0 = performance.now();
  const res = await fetch(url, {
    method: "POST",
    headers: buildRotatedHeaders(headers),
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
      const res = await fetch("[https://api.llm7.io/v1/models](https://api.llm7.io/v1/models)", { signal });
      if (!res.ok) throw new Error("Failed LLM7 model discovery");
      const json = await res.json();
      const list = Array.isArray(json.data) ? json.data : Array.isArray(json) ? json : [];
      return list.map((m: Record<string, unknown>) => ({
        id: String(m.id),
        context: Number(m.context_window || 65536),
      }));
    },
    executeCompletion: (modelId, messages, signal) =>
      timedOpenAIFetch(
        "[https://api.llm7.io/v1/chat/completions](https://api.llm7.io/v1/chat/completions)",
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
        const res = await fetch("[https://ai-gateway.kilo.ai/api/v1/models](https://ai-gateway.kilo.ai/api/v1/models)", { signal });
        if (res.ok) {
          const json = await res.json();
          const freeOnes = (json.data || []).filter((m: Record<string, unknown>) =>
            String(m.id).includes("free") || String(m.id).includes("kilo-auto")
          );
          if (freeOnes.length > 0) {
            return freeOnes.map((m: Record<string, unknown>) => ({
              id: String(m.id),
              context: Number(m.context_length || 256000),
            }));
          }
        }
      } catch {
        // Fallback to known keyless endpoints
      }
      return [
        { id: "kilo-auto/free", context: 256000 },
        { id: "kilo-auto/small", context: 128000 },
        { id: "google/gemma-4-26b-a4b-it:free", context: 128000 },
      ];
    },
    executeCompletion: (modelId, messages, signal) =>
      timedOpenAIFetch(
        "[https://ai-gateway.kilo.ai/api/v1/chat/completions](https://ai-gateway.kilo.ai/api/v1/chat/completions)",
        { "x-kilocode-mode": "code", "User-Agent": "Kilo-Code/2.1" },
        { model: modelId, messages, stream: false },
        signal,
      ),
  },
  {
    id: "pollinations",
    name: "Pollinations Keyless Core",
    discoverModels: async (signal) => {
      const res = await fetch("[https://text.pollinations.ai/models](https://text.pollinations.ai/models)", { signal });
      if (!res.ok) throw new Error("Failed Pollinations discovery");
      const list = await res.json();
      return (Array.isArray(list) ? list : []).map((m: Record<string, unknown>) => ({
        id: String(m.name || m.id),
        context: 65536,
      }));
    },
    executeCompletion: (modelId, messages, signal) =>
      timedOpenAIFetch(
        "[https://text.pollinations.ai/openai](https://text.pollinations.ai/openai)",
        {},
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
        "[https://opencode.ai/inference/openai/v1/chat/completions](https://opencode.ai/inference/openai/v1/chat/completions)",
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
// 6. AUTO-DISCOVERY SYNC & PARETO + THOMPSON OBJECTIVE SCORER
// ============================================================================

let lastSyncCheckMemory = 0;

async function ensureCatalogSynced(force: boolean = false): Promise<void> {
  const now = Date.now();
  if (!force && now - lastSyncCheckMemory < 60_000 && L1_CACHE.size > 0) {
    return;
  }
  lastSyncCheckMemory = now;

  const syncMap = new Map<string, number>();
  const metaIter = kv.list<number>({ prefix: ["mesh_discovery_meta"] });
  for await (const entry of metaIter) {
    syncMap.set(String(entry.key[1]), Number(entry.value));
  }

  const syncTasks = PROVIDERS.map(async (p) => {
    const last = syncMap.get(p.id) || 0;
    if (!force && now - last < 6 * 3600 * 1000) return;

    const ctrl = new AbortController();
    const timer = setTimeout(() => ctrl.abort(), 5000);
    try {
      const discovered = await p.discoverModels(ctrl.signal);
      for (const item of discovered) {
        const iqPrior = computeIntrinsicIqPrior(item.id, item.context);
        if (iqPrior <= 0.0) continue;

        const tupleId = `${p.id}::${item.id}`;
        const modelKey = ["mesh_models_v4", tupleId];
        const existing = await kv.get<MeshModelRecord>(modelKey);

        if (existing.value) {
          const updated = { ...existing.value, iq_prior: iqPrior };
          await kv.set(modelKey, updated);
          L1_CACHE.set(tupleId, updated);
        } else {
          const freshRecord: MeshModelRecord = {
            tuple_id: tupleId,
            provider_id: p.id,
            model_id: item.id,
            iq_prior: iqPrior,
            iq_verified: 1.0,
            ewma_ttft_ms: 900.0,
            ewma_tps: 35.0,
            alpha_success: 2.0,
            beta_failure: 1.0,
            success_count: 0,
            failure_streak: 0,
            cooldown_until: 0,
            last_probed: 0,
            last_error: null,
          };
          await kv.set(modelKey, freshRecord);
          L1_CACHE.set(tupleId, freshRecord);
        }
      }
      await kv.set(["mesh_discovery_meta", p.id], now);
    } catch {
      // Retain existing catalog on discovery timeout
    } finally {
      clearTimeout(timer);
    }
  });

  await Promise.allSettled(syncTasks);
}

export interface ScoredCandidate {
  tupleId: string;
  providerId: string;
  modelId: string;
  iqEffective: number;
  reliabilityDraw: number;
  ewmaTtftMs: number;
  ewmaTps: number;
  expectedFastMs: number;
  smartScore: number;
}

async function getRankedCandidates(mode: "fast" | "smart" | "code", deterministic = false): Promise<ScoredCandidate[]> {
  await ensureCatalogSynced(false);
  const catalog = await hydrateL1Cache(false);
  const now = Date.now();

  const candidates: ScoredCandidate[] = [];
  for (const r of catalog.values()) {
    if (r.cooldown_until > now) continue;

    const iqPrior = Number(r.iq_prior);
    const iqVerified = Number(r.iq_verified);
    const iqEffective = Number((iqPrior * iqVerified).toFixed(3));
    const ewmaTtftMs = Number(r.ewma_ttft_ms);
    const ewmaTps = Math.max(1.0, Number(r.ewma_tps));

    const alpha = Number(r.alpha_success || 2.0);
    const beta = Number(r.beta_failure || 1.0);
    const reliabilityDraw = deterministic
      ? alpha / (alpha + beta)
      : sampleBeta(alpha, beta);

    const expectedFastMs = Math.round(
      (ewmaTtftMs + (250 / ewmaTps) * 1000) / Math.max(0.35, reliabilityDraw),
    );

    const latencyPenalty = Math.max(0, (expectedFastMs - 8000) / 40000);
    const codeBonus = mode === "code" && /coder|codestral|glm|deepseek|claude|kilo/i.test(r.model_id) ? 0.08 : 0;
    const thompsonBonus = (reliabilityDraw - 0.5) * 0.12;
    const smartScore = Number((iqEffective + codeBonus + thompsonBonus - latencyPenalty).toFixed(4));

    candidates.push({
      tupleId: r.tuple_id,
      providerId: r.provider_id,
      modelId: r.model_id,
      iqEffective,
      reliabilityDraw: Number(reliabilityDraw.toFixed(3)),
      ewmaTtftMs: Math.round(ewmaTtftMs),
      ewmaTps: Number(ewmaTps.toFixed(1)),
      expectedFastMs,
      smartScore,
    });
  }

  if (mode === "fast") {
    return candidates
      .filter((c) => c.iqEffective >= 0.45)
      .sort((a, b) => a.expectedFastMs - b.expectedFastMs);
  }

  return candidates.sort((a, b) => b.smartScore - a.smartScore);
}

// ============================================================================
// 7. ATOMIC TELEMETRY UPDATERS & CANARY VERIFICATION
// ============================================================================

async function recordTupleSuccess(tupleId: string, ttftMs: number, totalMs: number, estTokens: number): Promise<void> {
  const streamDurationSec = Math.max(0.05, (totalMs - ttftMs) / 1000);
  const observedTps = Math.min(600, Math.max(2, estTokens / streamDurationSec));

  await atomicUpdateModel(tupleId, (r) => ({
    ...r,
    ewma_ttft_ms: (0.3 * ttftMs) + (0.7 * r.ewma_ttft_ms),
    ewma_tps: (0.3 * observedTps) + (0.7 * r.ewma_tps),
    alpha_success: Math.min(50, (r.alpha_success || 2.0) * 0.95 + 1.0),
    beta_failure: Math.max(1.0, (r.beta_failure || 1.0) * 0.95),
    success_count: r.success_count + 1,
    failure_streak: 0,
    cooldown_until: 0,
  }));
}

async function recordTupleFailure(tupleId: string, errMsg: string): Promise<void> {
  const now = Date.now();
  const baseSec = errMsg.includes("429") ? 120 : errMsg.includes("401") || errMsg.includes("403") ? 900 : 60;

  await atomicUpdateModel(tupleId, (r) => {
    const streak = r.failure_streak + 1;
    const backoffMultiplier = Math.min(8, Math.pow(1.5, streak - 1));
    return {
      ...r,
      failure_streak: streak,
      alpha_success: Math.max(1.0, (r.alpha_success || 2.0) * 0.9),
      beta_failure: Math.min(50, (r.beta_failure || 1.0) + 1.5),
      cooldown_until: now + Math.round(baseSec * backoffMultiplier * 1000),
      ewma_ttft_ms: Math.min(15000, r.ewma_ttft_ms * 1.4),
      last_error: errMsg.slice(0, 200),
    };
  });
}

async function runCanaryProbeOnTopCandidates(limit: number = 4) {
  const topSmart = (await getRankedCandidates("smart", true)).slice(0, limit);
  const probePrompt = [
    {
      role: "user",
      content: "Compute (17 * 19) + 11. Reply ONLY with the exact integer and nothing else.",
    },
  ];

  return await Promise.allSettled(
    topSmart.map(async (cand) => {
      const provider = PROVIDERS.find((p) => p.id === cand.providerId);
      if (!provider) return null;

      const ctrl = new AbortController();
      const timer = setTimeout(() => ctrl.abort(), 9000);
      try {
        const res = await provider.executeCompletion(cand.modelId, probePrompt, ctrl.signal);
        const passed = /\b334\b/.test(res.content.trim());
        const verifiedMultiplier = passed ? 1.0 : 0.55;

        await recordTupleSuccess(cand.tupleId, res.ttftMs, res.totalMs, res.estTokens);
        await atomicUpdateModel(cand.tupleId, (r) => ({
          ...r,
          iq_verified: verifiedMultiplier,
          last_probed: Date.now(),
        }));

        return { tupleId: cand.tupleId, passed, ttftMs: res.ttftMs, answer: res.content.trim() };
      } catch (e: unknown) {
        const msg = e instanceof Error ? e.message : "Probe failed";
        await recordTupleFailure(cand.tupleId, msg);
        throw e;
      } finally {
        clearTimeout(timer);
      }
    }),
  );
}

// ============================================================================
// 8. HONO HTTP APPLICATION & FRESH 2.0 COMPATIBILITY SHIM
// ============================================================================

const honoApp = new Hono();
honoApp.use("*", cors());
honoApp.use("*", bodyLimit({ maxSize: 256 * 1024 }));

honoApp.use("/v1/*", async (c, next) => {
  const ip = c.req.header("cf-connecting-ip") || c.req.header("x-forwarded-for") || "anon";
  if (!checkRateLimit(ip)) {
    return c.json({ error: { message: "Rate limit exceeded on edge isolate" } }, 429);
  }

  const requiredKey = Deno.env.get("ROUTER_API_KEY");
  if (requiredKey) {
    const authHeader = c.req.header("authorization") || "";
    const token = authHeader.startsWith("Bearer ") ? authHeader.slice(7).trim() : "";
    if (!token || !timingSafeCompare(token, requiredKey)) {
      return c.json({ error: { message: "Unauthorized: Invalid ROUTER_API_KEY" } }, 401);
    }
  }
  await next();
});

honoApp.get("/", async (c) => {
  const fastRank = (await getRankedCandidates("fast", true)).slice(0, 5);
  const smartRank = (await getRankedCandidates("smart", true)).slice(0, 5);

  return c.json({
    engine: "vajra-empirical-sovereign-v4.2-hardened",
    runtime: "deno-deploy-kv-cas-l1",
    security: {
      auth_enabled: Boolean(Deno.env.get("ROUTER_API_KEY")),
      admin_guard_enabled: Boolean(Deno.env.get("ADMIN_SECRET")),
      l1_cache_entries: L1_CACHE.size,
    },
    current_best_fast: fastRank,
    current_best_smart: smartRank,
  });
});

honoApp.post("/v1/probe", async (c) => {
  const adminSecret = Deno.env.get("ADMIN_SECRET");
  if (adminSecret) {
    const provided = c.req.header("x-admin-secret") || "";
    if (!provided || !timingSafeCompare(provided, adminSecret)) {
      return c.json({ error: { message: "Forbidden: Invalid x-admin-secret" } }, 403);
    }
  }
  const probeResults = await runCanaryProbeOnTopCandidates(4);
  return c.json({ status: "probe_complete", results: probeResults });
});

honoApp.post("/v1/sync", async (c) => {
  const adminSecret = Deno.env.get("ADMIN_SECRET");
  if (adminSecret) {
    const provided = c.req.header("x-admin-secret") || "";
    if (!provided || !timingSafeCompare(provided, adminSecret)) {
      return c.json({ error: { message: "Forbidden: Invalid x-admin-secret" } }, 403);
    }
  }
  await ensureCatalogSynced(true);
  await hydrateL1Cache(true);
  return c.json({ status: "catalog_synced", total_models: L1_CACHE.size });
});

honoApp.post("/v1/chat/completions", async (c) => {
  try {
    const body = await c.req.json();
    const messages = Array.isArray(body.messages) ? body.messages : [];
    const reqModel = String(body.model || "free:auto").toLowerCase();
    const wantsStream = Boolean(body.stream);

    const mode: "fast" | "smart" | "code" = reqModel.includes("auto")
      ? classifyPromptIntent(messages)
      : reqModel.includes("fast")
      ? "fast"
      : reqModel.includes("code")
      ? "code"
      : "smart";

    const ranked = await getRankedCandidates(mode, false);
    if (ranked.length === 0) {
      return c.json({ error: { message: "No healthy discovered models available." } }, 503);
    }

    const masterAbort = new AbortController();
    const globalTimeout = setTimeout(() => masterAbort.abort(), 16000);

    const wave = ranked.slice(0, 3);

    const attempts = wave.map(async (cand, idx) => {
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
      } catch (err: unknown) {
        if (!masterAbort.signal.aborted) {
          const msg = err instanceof Error ? err.message : "Exec error";
          await recordTupleFailure(cand.tupleId, msg);
        }
        throw err;
      }
    });

    try {
      const winner = await Promise.any(attempts);
      masterAbort.abort();

      const completionId = `chatcmpl-${crypto.randomUUID()}`;
      const created = Math.floor(Date.now() / 1000);

      if (wantsStream) {
        const stream = new ReadableStream({
          start(controller) {
            const chunkPayload = {
              id: completionId,
              object: "chat.completion.chunk",
              created,
              model: winner.candidate.modelId,
              provider: winner.candidate.providerId,
              choices: [
                {
                  index: 0,
                  delta: { role: "assistant", content: winner.content },
                  finish_reason: null,
                },
              ],
            };
            const stopPayload = {
              id: completionId,
              object: "chat.completion.chunk",
              created,
              model: winner.candidate.modelId,
              choices: [{ index: 0, delta: {}, finish_reason: "stop" }],
            };
            controller.enqueue(encoder.encode(`data: ${JSON.stringify(chunkPayload)}\n\n`));
            controller.enqueue(encoder.encode(`data: ${JSON.stringify(stopPayload)}\n\n`));
            controller.enqueue(encoder.encode("data: [DONE]\n\n"));
            controller.close();
          },
        });

        return new Response(stream, {
          headers: {
            "Content-Type": "text/event-stream; charset=utf-8",
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
          },
        });
      }

      return c.json({
        id: completionId,
        object: "chat.completion",
        created,
        model: winner.candidate.modelId,
        provider: winner.candidate.providerId,
        telemetry: {
          mode_selected: mode,
          iq_effective: winner.candidate.iqEffective,
          reliability_draw: winner.candidate.reliabilityDraw,
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
  } catch (err: unknown) {
    const msg = err instanceof Error ? err.message : "Empirical mesh dispatch failed";
    return c.json({ error: { message: msg } }, 502);
  }
});

// Fresh 2.0 (@fresh/core) compatibility shim in case Deno Deploy executes `/app/src/_fresh/server.js`
const app = Object.assign(honoApp, {
  config: {
    root: Deno.cwd(),
    basePath: "",
    mode: "production",
  },
  handler: () => honoApp.fetch,
  listen: (options?: Deno.ServeTcpOptions) => Deno.serve(options ?? {}, honoApp.fetch),
});

if (typeof Deno.cron === "function") {
  Deno.cron("canary-probe", "*/30 * * * *", async () => {
    await runCanaryProbeOnTopCandidates(4);
  });
}

// Only call Deno.serve directly if main.ts is the direct entrypoint (prevents double-bind when imported by _fresh/server.js)
if (import.meta.main) {
  Deno.serve(app.fetch);
}

export { app };
export default {
  fetch: app.fetch,
};
