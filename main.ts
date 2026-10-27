import { Hono } from "hono";
import { cors } from "hono/cors";
import { bodyLimit } from "hono/body-limit";

// ============================================================================
// 1. DENO KV + L1 ISOLATE CACHE + CAS ATOMIC STATE ENGINE
// ============================================================================

const kv = await Deno.openKv();
const KV_PREFIX = "mesh_models_v5_2";

export interface MeshModelRecord {
  tuple_id: string;
  provider_id: string;
  model_id: string;
  iq_prior: number;
  iq_verified: number;
  supports_reasoning: boolean;
  supports_vision: boolean;
  is_coding_specialist: boolean;
  context_window: number;
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
  const iter = kv.list<MeshModelRecord>({ prefix: [KV_PREFIX] });
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
  const key = [KV_PREFIX, tupleId];
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
// 2. SECURITY: CONSTANT-TIME AUTH, RATE LIMITER & HEADER ROTATOR
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
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/142.0.0.0 Safari/537.36",
    "OpenAI/v1 PythonBindings/1.82.0",
    "continue-dev/1.4.12 (darwin-arm64)",
    "cline/4.1.0 vscode/1.105.0",
    "roo-code/3.18.0 (win32-x64)",
    "opencode-ai/2.6.4 (linux-x64)",
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

function pickOpenRouterKey(): string | null {
  const raw = Deno.env.get("OPENROUTER_API_KEY") || Deno.env.get("OPENROUTER_API_KEYS") || "";
  const keys = raw.split(",").map((k) => k.trim()).filter(Boolean);
  if (keys.length === 0) return null;
  return keys[Math.floor(Math.random() * keys.length)];
}

// ============================================================================
// 3. VERIFIED OCT-2026 CAPABILITY PARSER & SHANNON ENTROPY POISON GUARD
// ============================================================================

interface ModelCapabilityProfile {
  iqPrior: number;
  supportsReasoning: boolean;
  supportsVision: boolean;
  isCodingSpecialist: boolean;
}

function analyzeModelCapabilities(modelId: string, contextLength: number = 128000): ModelCapabilityProfile {
  const m = modelId.toLowerCase();

  // Exclude non-chat, moderation/guardrail, audio-only, image-gen, and embedding endpoints
  // Note: nvidia/nemotron-3.5-content-safety:free is a guardrail moderation classifier, not a general chat LLM
  if (
    /embed|bge-|e5-|whisper|tts|kokoro|flux|sdxl|dall-e|imagen|veo-|moderation|guard|content-safety|rerank|suno|udio|respan/i
      .test(m)
  ) {
    return { iqPrior: 0.0, supportsReasoning: false, supportsVision: false, isCodingSpecialist: false };
  }

  let baseScore = 0.56;

  // Tier 0: October 2026 Apex Frontier (AA Intelligence Index 52.5 - 57.6 / Vals #1)
  // Includes: Claude Opus 5.5 (57.6), Claude Sonnet 5.5 (56.0), Kimi K3.1 Pro (55.1, 2.8T/104B),
  // Claude Fable 5.1 (53.4), Gemini 4 Argon (52.6, #1 Vals Index), GPT-6.1 Terra, GPT-6 Astra
  if (
    /claude-opus-5(\.5)?|claude-sonnet-5(\.5)?|claude-fable-5(\.1)?|kimi-k3(\.1)?-pro|gemini-4-argon|argon|gpt-6(\.1)?-(terra|astra)|space-bunny-alpha/i
      .test(m)
  ) {
    baseScore = 0.975;
  } // Tier 1: Late-2026 Flagships & Ultra-Heavy Free MoEs
  // Includes: Grok 4.7 / 4.5 / 4.4 / 4-Heavy, Gemini 3.1 Pro (77.1% ARC-AGI-2), Gemini 3.8 Flash, Gemini 2.8,
  // Qwen3.8 Max, DeepSeek V4 Pro (1.6T/49B), DeepSeek V4.1 Flash (1017 CED), GLM 5.3,
  // ThinkingMachines Inkling (975B/41B :free), NVIDIA Nemotron 3 Ultra (550B/55B :free),
  // Poolside Laguna S 2.1 (118B/8B :free), Dots3-Note Preview (280B/16B :free)
  else if (
    /grok-4(\.[13457]|-heavy)?(?!-mini)|gemini-(3\.[158]|2\.8)-(pro|flash)|qwen3\.8-max|deepseek-v4(\.1)?-(pro|flash)|glm-5\.3|thinkingmachines\/inkling(?!-small)|nemotron-3-ultra-550b|laguna-s-2\.1|dots-3-note|claude-4-[57]|claude-opus-4|gpt-5\.6-sol|o3(?!-mini)|o4(?!-mini)/i
      .test(m)
  ) {
    baseScore = 0.935;
  } // Tier 2: High-Efficiency Agentic, Coding & Multimodal Workhorses (Oct 2026 :free core)
  // Includes: GPT-6 Luna, GPT-5.6 Luna, Kimi K3 / K2, DeepSeek V4 Flash 0731, DeepSeek V3.2, DeepSeek R1 0528,
  // Inkling Small (276B/12B :free), Nemotron 3 Super (120B/12B :free), Qwen3.8 27B (:free),
  // Gemma 4 31B IT (:free), Gemma 4 26B A4B (:free), Cohere North Mini Code (30B/3B :free),
  // Poolside Laguna XS 2.1 (:free), Recursive Flint (40B/2.5B :free), Xing4.0 29B A4B (:free),
  // MiMo-V2.6-Flash, Tencent Hy4 Preview, MiniMax M2, OpenRouter Auto-Free Router (openrouter/free)
  else if (
    /gpt-6-luna|gpt-5\.6-luna|kimi-k[23]|deepseek-v4-flash|deepseek-v3\.2|deepseek-r1|inkling-small|nemotron-3-super-120b|qwen3\.8-27b|qwen3(\.5)?-(32b|72b|235b|coder)|gemma-?4-(31b|26b)|north-mini-code|laguna-xs-2\.1|recursive\/flint|xing4\.0-29b|mimo-v2\.6|hy4-preview|minimax-m2|ling-3\.0-flash|solar-pro-4|openrouter\/free|kilo-auto\/free|claude-3-[57]-sonnet|gpt-4\.1(?!-nano|-mini)|llama-4-(maverick|behemoth)/i
      .test(m)
  ) {
    baseScore = 0.875;
  } // Tier 3: High-Throughput Agent Lightning & Perception Sub-Agents
  // Includes: Nemotron 3.5 Lightning (30B/3B :free, 1M ctx), Nemotron 3 Nano Omni (30B/3B :free),
  // Nemotron 3.5 Bolt E4B, Apodex 1.1 Mini (:free), Gemma 3 27B, Llama 3.3 70B, Llama 4 Scout
  else if (
    /nemotron-3\.5-lightning|nemotron-3-nano-omni|nemotron-3\.5-bolt|apodex-1\.1-mini|gemma-?3-27b|llama-3\.3-70b|llama-4-scout|qwen-?2\.5-(coder-32b|72b)|codestral|mistral-large|glm-4\.5|grok-4-mini|grok-3|gpt-4o-mini|claude-3-5-haiku|kilo-auto\/small/i
      .test(m)
  ) {
    baseScore = 0.77;
  } // Tier 4: Compact Extraction / Edge 2B-9B Models
  // Note: Liquid LFM2.5-2.6B (:free) is explicitly advised by Liquid AI for RAG/extraction, NOT agentic coding
  else if (
    /lfm-2\.5-2\.6b|gemma-?[34]-(9b|12b)|qwen3-(8b|4b)|llama-3\.[12]-(8b|11b)|ministral|mistral-nemo|phi-4|flash-lite/i
      .test(m)
  ) {
    baseScore = 0.52;
  } // Tier 5: Sub-2B Micro Models
  else if (/0\.5b|0\.6b|1b|1\.5b|tinyllama|smollm/i.test(m)) {
    baseScore = 0.35;
  }

  const supportsReasoning =
    /argon|claude-opus-5|claude-sonnet-5|claude-fable-5|kimi-k3|gpt-6|grok-4|gemini-3|gemini-2\.8|deepseek-v4|deepseek-v3\.2|deepseek-r1|glm-5|qwen3\.8|inkling|nemotron-3|dots-3|xing4\.0|apodex|lfm-2\.5|o1|o3|o4|thinking|reason/i
      .test(m);

  const supportsVision =
    /argon|claude-opus-5|claude-sonnet-5|kimi-k3|gpt-6|grok-4|gemini|qwen3\.8|gemma-?4-(26b|31b)|inkling|nemotron-3-nano-omni|dots-3|deepseek-v3\.2|vision|vl|pixtral|llama-4/i
      .test(m);

  const isCodingSpecialist =
    /laguna|north-mini-code|flint|coder|codestral|devstral|kimi-k3|claude-opus-5|claude-sonnet-5|gemini-4-argon|deepseek-v4|glm-5\.3|qwen3\.8|inkling|kilo/i
      .test(m);

  if (isCodingSpecialist) baseScore += 0.015;
  if (contextLength >= 256000) baseScore += 0.01;
  if (contextLength >= 1000000) baseScore += 0.01;

  return {
    iqPrior: Math.min(0.995, Number(baseScore.toFixed(3))),
    supportsReasoning,
    supportsVision,
    isCodingSpecialist,
  };
}

const POISON_REGEX =
  /rate limit exceeded|too many requests|cloudflare ray id|just a moment\.\.\.|api key is required|insufficient_quota|error code: 1020|visit https:\/\/|upgrade your plan|free tier limit|model is overloaded|service unavailable|provider returned error/i;

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

function splitThinkBlock(raw: string): { content: string; reasoning: string | null } {
  const thinkMatch = raw.match(/^[\s]*<think>([\s\S]*?)<\/think>([\s\S]*)$/i);
  if (thinkMatch) {
    const reasoning = thinkMatch[1].trim();
    const content = thinkMatch[2].trim();
    return { content: content || reasoning, reasoning: reasoning || null };
  }
  return { content: raw, reasoning: null };
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
// 4. EPISTEMIC THOMPSON SAMPLING + MULTIMODAL INTENT AUTO-CLASSIFIER
// ============================================================================

export type RoutingMode = "fast" | "smart" | "code" | "reasoning";

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

function extractMessageText(messages: Array<{ role: string; content: unknown }>): {
  combinedText: string;
  hasVisionPayload: boolean;
} {
  let hasVisionPayload = false;
  const parts: string[] = [];

  for (const m of messages) {
    if (typeof m.content === "string") {
      parts.push(m.content);
    } else if (Array.isArray(m.content)) {
      for (const block of m.content) {
        if (block && typeof block === "object") {
          const b = block as Record<string, unknown>;
          if (b.type === "text" && typeof b.text === "string") {
            parts.push(b.text);
          } else if (
            b.type === "image_url" ||
            b.type === "image" ||
            b.type === "input_image" ||
            b.type === "video_url" ||
            b.type === "input_audio"
          ) {
            hasVisionPayload = true;
          }
        }
      }
    }
  }
  return { combinedText: parts.join("\n"), hasVisionPayload };
}

function classifyPromptIntent(
  messages: Array<{ role: string; content: unknown }>,
  hasTools = false,
  reasoningEffort?: string,
): RoutingMode {
  if (reasoningEffort === "high" || reasoningEffort === "xhigh" || reasoningEffort === "max") {
    return "reasoning";
  }
  const { combinedText } = extractMessageText(messages);

  if (
    hasTools ||
    /```|function\s+\w+|interface\s+\w+|def\s+\w+|import\s+\{|SELECT\s+.*\s+FROM|cargo\s+build|Deno\.|async\s+function|pub\s+fn\s+|#include\s*</i
      .test(combinedText)
  ) {
    return "code";
  }
  if (
    /prove\s+that|formal\s+proof|theorem|lemma|eigenvalue|hamiltonian|lagrangian|step-by-step\s+derivation|game\s+theory|bayesian|time\s+complexity\s+proof/i
      .test(combinedText)
  ) {
    return "reasoning";
  }
  if (combinedText.length > 1800 || /architect|analyze|differential|rigorous|synthesize|tradeoff/i.test(combinedText)) {
    return "smart";
  }
  return "fast";
}

// ============================================================================
// 5. LIVE UPSTREAM MODEL CATALOG DISCOVERY ADAPTERS (INCL. OPENROUTER :free)
// ============================================================================

interface CompletionResult {
  content: string;
  reasoningContent: string | null;
  toolCalls?: unknown[];
  ttftMs: number;
  totalMs: number;
  estTokens: number;
}

interface CompletionOptions {
  temperature?: number;
  max_tokens?: number;
  top_p?: number;
  reasoning_effort?: string;
  tools?: unknown[];
  tool_choice?: unknown;
  response_format?: unknown;
}

interface ProviderSpec {
  id: string;
  name: string;
  isConfigured: () => boolean;
  discoverModels: (signal: AbortSignal) => Promise<Array<{ context?: id: number string; }>>;
  executeCompletion: (
    modelId: string,
    messages: Array<{ role: string; content: unknown }>,
    signal: AbortSignal,
    options?: CompletionOptions,
  ) => Promise<CompletionResult>;
}

async function timedOpenAIFetch(
  url: string,
  headers: Record<string, string>,
  payload: Record<string, unknown>,
  signal: AbortSignal,
): Promise<CompletionResult> {
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

  let rawContent = "";
  let rawReasoning = "";
  let toolCalls: unknown[] | undefined = undefined;

  if (rawText.startsWith("data: ")) {
    for (const line of rawText.split("\n")) {
      if (line.startsWith("data: ")) {
        const slice = line.slice(6).trim();
        if (slice === "[DONE]") continue;
        try {
          const parsed = JSON.parse(slice);
          const delta = parsed.choices?.[0]?.delta ?? parsed.choices?.[0]?.message ?? {};
          rawContent += delta.content ?? "";
          rawReasoning += delta.reasoning_content ?? delta.reasoning ?? "";
          if (Array.isArray(delta.tool_calls)) {
            toolCalls = delta.tool_calls;
          }
        } catch {
          // Ignore keepalive frames
        }
      }
    }
  } else {
    const data = JSON.parse(rawText);
    if (data.error) {
      throw new Error(`UPSTREAM_ERR: ${data.error.message || JSON.stringify(data.error)}`);
    }
    const msg = data.choices?.[0]?.message ?? {};
    rawContent = msg.content ?? data.choices?.[0]?.text ?? data.response ?? "";
    rawReasoning = msg.reasoning_content ?? msg.reasoning ?? "";
    if (Array.isArray(msg.tool_calls) && msg.tool_calls.length > 0) {
      toolCalls = msg.tool_calls;
    }
  }

  const { content, reasoning } = splitThinkBlock(rawContent);
  const finalReasoning = rawReasoning.trim() || reasoning || null;

  if (!toolCalls || toolCalls.length === 0) {
    verifyCleanText(content);
  }

  const totalChars = content.length + (finalReasoning?.length ?? 0);
  const estTokens = Math.max(1, Math.ceil(totalChars / 4));

  return {
    content,
    reasoningContent: finalReasoning,
    toolCalls,
    ttftMs: ttftHeaderMs,
    totalMs,
    estTokens,
  };
}

function buildOptionalParams(options?: CompletionOptions): Record<string, unknown> {
  if (!options) return {};
  const extra: Record<string, unknown> = {};
  if (typeof options.temperature === "number") extra.temperature = options.temperature;
  if (typeof options.max_tokens === "number") extra.max_tokens = options.max_tokens;
  if (typeof options.top_p === "number") extra.top_p = options.top_p;
  if (options.reasoning_effort) extra.reasoning_effort = options.reasoning_effort;
  if (Array.isArray(options.tools) && options.tools.length > 0) extra.tools = options.tools;
  if (options.tool_choice !== undefined) extra.tool_choice = options.tool_choice;
  if (options.response_format !== undefined) extra.response_format = options.response_format;
  return extra;
}

// Verified October 2026 OpenRouter :free model catalog fallback
const OPENROUTER_OCT2026_FREE_FALLBACK: Array<{ id: string; context: number }> = [
  { id: "openrouter/free", context: 262144 },
  { id: "thinkingmachines/inkling:free", context: 1048576 },
  { id: "thinkingmachines/inkling-small:free", context: 1048576 },
  { id: "nvidia/nemotron-3-ultra-550b-a55b:free", context: 1048576 },
  { id: "nvidia/nemotron-3-super-120b-a12b:free", context: 262144 },
  { id: "nvidia/nemotron-3.5-lightning:free", context: 1048576 },
  { id: "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free", context: 262144 },
  { id: "poolside/laguna-s-2.1:free", context: 262144 },
  { id: "poolside/laguna-xs-2.1:free", context: 262144 },
  { id: "cohere/north-mini-code:free", context: 256000 },
  { id: "recursive/flint:free", context: 262144 },
  { id: "qwen/qwen3.8-27b:free", context: 262144 },
  { id: "google/gemma-4-31b-it:free", context: 262144 },
  { id: "google/gemma-4-26b-a4b-it:free", context: 262144 },
  { id: "dots-studio/dots-3-note-preview:free", context: 524288 },
  { id: "xingchen-agi/xing4.0-29b-a4b:free", context: 262144 },
  { id: "apodex/apodex-1.1-mini:free", context: 262144 },
  { id: "inclusionai/ling-3.0-flash-sante:free", context: 262144 },
  { id: "liquid/lfm-2.5-2.6b:free", context: 65536 },
];

const PROVIDERS: ProviderSpec[] = [
  {
    id: "openrouter",
    name: "OpenRouter Verified :free Mesh",
    isConfigured: () => Boolean(pickOpenRouterKey()),
    discoverModels: async (signal) => {
      try {
        // OpenRouter's /api/v1/models endpoint is public and returns live pricing & :free slugs
        const res = await fetch("[https://openrouter.ai/api/v1/models](https://openrouter.ai/api/v1/models)", { signal });
        if (res.ok) {
          const json = await res.json();
          const data: Array<Record<string, unknown>> = Array.isArray(json.data) ? json.data : [];
          const freeModels = data.filter((m) => {
            const id = String(m.id || "");
            const pricing = (m.pricing as Record<string, unknown>) || {};
            const isZeroCost =
              String(pricing.prompt) === "0" && String(pricing.completion) === "0";
            return id.endsWith(":free") || id === "openrouter/free" || isZeroCost;
          });

          if (freeModels.length > 0) {
            const mapped = freeModels.map((m) => ({
              id: String(m.id),
              context: Number(m.context_length || 262144),
            }));
            if (!mapped.some((m) => m.id === "openrouter/free")) {
              mapped.unshift({ id: "openrouter/free", context: 262144 });
            }
            return mapped;
          }
        }
      } catch {
        // Fallback to verified October 2026 :free list
      }
      return OPENROUTER_OCT2026_FREE_FALLBACK;
    },
    executeCompletion: (modelId, messages, signal, options) => {
      const apiKey = pickOpenRouterKey();
      if (!apiKey) {
        throw new Error("HTTP_401: OPENROUTER_API_KEY not configured for openrouter provider");
      }
      return timedOpenAIFetch(
        "[https://openrouter.ai/api/v1/chat/completions](https://openrouter.ai/api/v1/chat/completions)",
        {
          "Authorization": `Bearer ${apiKey}`,
          "HTTP-Referer": "[https://vajra-mesh.deno.dev](https://vajra-mesh.deno.dev)",
          "X-Title": "Vajra Empirical Sovereign Router",
        },
        { model: modelId, messages, stream: false, ...buildOptionalParams(options) },
        signal,
      );
    },
  },
  {
    id: "llm7",
    name: "LLM7 Anonymous Mesh",
    isConfigured: () => true,
    discoverModels: async (signal) => {
      const res = await fetch("[https://api.llm7.io/v1/models](https://api.llm7.io/v1/models)", { signal });
      if (!res.ok) throw new Error("Failed LLM7 model discovery");
      const json = await res.json();
      const list = Array.isArray(json.data) ? json.data : Array.isArray(json) ? json : [];
      return list.map((m: Record<string, unknown>) => ({
        id: String(m.id),
        context: Number(m.context_window || m.context_length || 128000),
      }));
    },
    executeCompletion: (modelId, messages, signal, options) =>
      timedOpenAIFetch(
        "[https://api.llm7.io/v1/chat/completions](https://api.llm7.io/v1/chat/completions)",
        { "Authorization": "Bearer unused" },
        { model: modelId, messages, stream: false, ...buildOptionalParams(options) },
        signal,
      ),
  },
  {
    id: "kilo",
    name: "Kilo Code AI Gateway",
    isConfigured: () => true,
    discoverModels: async (signal) => {
      try {
        const res = await fetch("[https://ai-gateway.kilo.ai/api/v1/models](https://ai-gateway.kilo.ai/api/v1/models)", { signal });
        if (res.ok) {
          const json = await res.json();
          const freeOnes = (json.data || []).filter((m: Record<string, unknown>) =>
            String(m.id).includes(":free") ||
            String(m.id).includes("free") ||
            String(m.id).includes("kilo-auto")
          );
          if (freeOnes.length > 0) {
            return freeOnes.map((m: Record<string, unknown>) => ({
              id: String(m.id),
              context: Number(m.context_length || 262144),
            }));
          }
        }
      } catch {
        // Fallback to known keyless endpoints
      }
      return [
        { id: "kilo-auto/free", context: 262144 },
        { id: "kilo-auto/small", context: 128000 },
        { id: "google/gemma-4-31b-it:free", context: 262144 },
        { id: "google/gemma-4-26b-a4b-it:free", context: 262144 },
        { id: "qwen/qwen3.8-27b:free", context: 262144 },
        { id: "poolside/laguna-s-2.1:free", context: 262144 },
        { id: "cohere/north-mini-code:free", context: 256000 },
      ];
    },
    executeCompletion: (modelId, messages, signal, options) =>
      timedOpenAIFetch(
        "[https://ai-gateway.kilo.ai/api/v1/chat/completions](https://ai-gateway.kilo.ai/api/v1/chat/completions)",
        { "x-kilocode-mode": "code", "User-Agent": "Kilo-Code/3.4.0" },
        { model: modelId, messages, stream: false, ...buildOptionalParams(options) },
        signal,
      ),
  },
  {
    id: "pollinations",
    name: "Pollinations Keyless Core",
    isConfigured: () => true,
    discoverModels: async (signal) => {
      const res = await fetch("[https://text.pollinations.ai/models](https://text.pollinations.ai/models)", { signal });
      if (!res.ok) throw new Error("Failed Pollinations discovery");
      const list = await res.json();
      return (Array.isArray(list) ? list : []).map((m: Record<string, unknown>) => ({
        id: String(m.name || m.id),
        context: Number(m.context_length || m.maxInputChars || 128000),
      }));
    },
    executeCompletion: (modelId, messages, signal, options) =>
      timedOpenAIFetch(
        "[https://text.pollinations.ai/openai](https://text.pollinations.ai/openai)",
        {},
        { model: modelId, messages, stream: false, ...buildOptionalParams(options) },
        signal,
      ),
  },
  {
    id: "opencode",
    name: "OpenCode V2 Zen",
    isConfigured: () => true,
    discoverModels: async () => [
      { id: "deepseek-v4.1-flash-free", context: 1048576 },
      { id: "glm-5.3-flash-free", context: 1048576 },
      { id: "nemotron-3.5-lightning-free", context: 1048576 },
      { id: "qwen3.8-27b-free", context: 262144 },
      { id: "mimo-v2.6-flash-free", context: 262144 },
    ],
    executeCompletion: (modelId, messages, signal, options) =>
      timedOpenAIFetch(
        "[https://opencode.ai/inference/openai/v1/chat/completions](https://opencode.ai/inference/openai/v1/chat/completions)",
        {
          "x-opencode-client": "cli",
          "x-opencode-project": "opencode-cli",
          "User-Agent": "opencode-ai/2.6.4",
        },
        { model: modelId, messages, stream: false, ...buildOptionalParams(options) },
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
  const metaIter = kv.list<number>({ prefix: ["mesh_discovery_meta_v5_2"] });
  for await (const entry of metaIter) {
    syncMap.set(String(entry.key[1]), Number(entry.value));
  }

  const syncTasks = PROVIDERS.map(async (p) => {
    if (!p.isConfigured()) return;
    const last = syncMap.get(p.id) || 0;
    if (!force && now - last < 6 * 3600 * 1000) return;

    const ctrl = new AbortController();
    const timer = setTimeout(() => ctrl.abort(), 5500);
    try {
      const discovered = await p.discoverModels(ctrl.signal);
      for (const item of discovered) {
        const ctxWindow = item.context || 128000;
        const profile = analyzeModelCapabilities(item.id, ctxWindow);
        if (profile.iqPrior <= 0.0) continue;

        const tupleId = `${p.id}::${item.id}`;
        const modelKey = [KV_PREFIX, tupleId];
        const existing = await kv.get<MeshModelRecord>(modelKey);

        if (existing.value) {
          const updated: MeshModelRecord = {
            ...existing.value,
            iq_prior: profile.iqPrior,
            supports_reasoning: profile.supportsReasoning,
            supports_vision: profile.supportsVision,
            is_coding_specialist: profile.isCodingSpecialist,
            context_window: ctxWindow,
          };
          await kv.set(modelKey, updated);
          L1_CACHE.set(tupleId, updated);
        } else {
          const freshRecord: MeshModelRecord = {
            tuple_id: tupleId,
            provider_id: p.id,
            model_id: item.id,
            iq_prior: profile.iqPrior,
            iq_verified: 1.0,
            supports_reasoning: profile.supportsReasoning,
            supports_vision: profile.supportsVision,
            is_coding_specialist: profile.isCodingSpecialist,
            context_window: ctxWindow,
            ewma_ttft_ms: 900.0,
            ewma_tps: 42.0,
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
      await kv.set(["mesh_discovery_meta_v5_2", p.id], now);
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
  supportsReasoning: boolean;
  supportsVision: boolean;
  isCodingSpecialist: boolean;
  contextWindow: number;
  reliabilityDraw: number;
  ewmaTtftMs: number;
  ewmaTps: number;
  expectedFastMs: number;
  smartScore: number;
}

async function getRankedCandidates(
  mode: RoutingMode,
  deterministic = false,
  requireVision = false,
): Promise<ScoredCandidate[]> {
  await ensureCatalogSynced(false);
  const catalog = await hydrateL1Cache(false);
  const now = Date.now();

  const activeProviderIds = new Set(PROVIDERS.filter((p) => p.isConfigured()).map((p) => p.id));

  const candidates: ScoredCandidate[] = [];
  for (const r of catalog.values()) {
    if (!activeProviderIds.has(r.provider_id)) continue;
    if (r.cooldown_until > now) continue;
    if (requireVision && !r.supports_vision) continue;

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

    const latencyPenalty = Math.max(0, (expectedFastMs - 8500) / 42000);
    // Liquid AI explicitly advises against using LFM-2.5-2.6B for agentic coding
    const isLfmSmall = /lfm-2\.5-2\.6b/i.test(r.model_id);
    const codeBonus = mode === "code"
      ? (r.is_coding_specialist ? 0.085 : isLfmSmall ? -0.25 : 0)
      : 0;
    const reasoningBonus = mode === "reasoning" && r.supports_reasoning ? 0.095 : 0;
    const thompsonBonus = (reliabilityDraw - 0.5) * 0.12;
    const smartScore = Number(
      (iqEffective + codeBonus + reasoningBonus + thompsonBonus - latencyPenalty).toFixed(4),
    );

    candidates.push({
      tupleId: r.tuple_id,
      providerId: r.provider_id,
      modelId: r.model_id,
      iqEffective,
      supportsReasoning: Boolean(r.supports_reasoning),
      supportsVision: Boolean(r.supports_vision),
      isCodingSpecialist: Boolean(r.is_coding_specialist),
      contextWindow: Number(r.context_window || 128000),
      reliabilityDraw: Number(reliabilityDraw.toFixed(3)),
      ewmaTtftMs: Math.round(ewmaTtftMs),
      ewmaTps: Number(ewmaTps.toFixed(1)),
      expectedFastMs,
      smartScore,
    });
  }

  if (mode === "fast") {
    return candidates
      .filter((c) => c.iqEffective >= 0.50)
      .sort((a, b) => a.expectedFastMs - b.expectedFastMs);
  }

  return candidates.sort((a, b) => b.smartScore - a.smartScore);
}

// ============================================================================
// 7. ATOMIC TELEMETRY UPDATERS & CANARY VERIFICATION
// ============================================================================

async function recordTupleSuccess(
  tupleId: string,
  ttftMs: number,
  totalMs: number,
  estTokens: number,
): Promise<void> {
  const streamDurationSec = Math.max(0.05, (totalMs - ttftMs) / 1000);
  const observedTps = Math.min(800, Math.max(2, estTokens / streamDurationSec));

  await atomicUpdateModel(tupleId, (r) => ({
    ...r,
    ewma_ttft_ms: 0.3 * ttftMs + 0.7 * r.ewma_ttft_ms,
    ewma_tps: 0.3 * observedTps + 0.7 * r.ewma_tps,
    alpha_success: Math.min(50, (r.alpha_success || 2.0) * 0.95 + 1.0),
    beta_failure: Math.max(1.0, (r.beta_failure || 1.0) * 0.95),
    success_count: r.success_count + 1,
    failure_streak: 0,
    cooldown_until: 0,
  }));
}

async function recordTupleFailure(tupleId: string, errMsg: string): Promise<void> {
  const now = Date.now();
  const baseSec = errMsg.includes("429")
    ? 120
    : errMsg.includes("401") || errMsg.includes("403")
    ? 900
    : 60;

  await atomicUpdateModel(tupleId, (r) => {
    const streak = r.failure_streak + 1;
    const backoffMultiplier = Math.min(8, Math.pow(1.5, streak - 1));
    return {
      ...r,
      failure_streak: streak,
      alpha_success: Math.max(1.0, (r.alpha_success || 2.0) * 0.9),
      beta_failure: Math.min(50, (r.beta_failure || 1.0) + 1.5),
      cooldown_until: now + Math.round(baseSec * backoffMultiplier * 1000),
      ewma_ttft_ms: Math.min(18000, r.ewma_ttft_ms * 1.4),
      last_error: errMsg.slice(0, 200),
    };
  });
}

async function runCanaryProbeOnTopCandidates(limit: number = 4) {
  const topSmart = (await getRankedCandidates("smart", true)).slice(0, limit);
  const probePrompt = [
    {
      role: "user",
      content: "Compute (23 * 19) + 14. Reply ONLY with the exact integer and nothing else.",
    },
  ];

  return await Promise.allSettled(
    topSmart.map(async (cand) => {
      const provider = PROVIDERS.find((p) => p.id === cand.providerId);
      if (!provider || !provider.isConfigured()) return null;

      const ctrl = new AbortController();
      const timer = setTimeout(() => ctrl.abort(), 11000);
      try {
        const res = await provider.executeCompletion(cand.modelId, probePrompt, ctrl.signal);
        const passed = /\b451\b/.test(res.content.trim());
        const verifiedMultiplier = passed ? 1.0 : 0.55;

        await recordTupleSuccess(cand.tupleId, res.ttftMs, res.totalMs, res.estTokens);
        await atomicUpdateModel(cand.tupleId, (r) => ({
          ...r,
          iq_verified: verifiedMultiplier,
          last_probed: Date.now(),
        }));

        return {
          tupleId: cand.tupleId,
          passed,
          ttftMs: res.ttftMs,
          hasReasoning: Boolean(res.reasoningContent),
          answer: res.content.trim().slice(0, 80),
        };
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
honoApp.use("*", bodyLimit({ maxSize: 512 * 1024 }));

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
  const codeRank = (await getRankedCandidates("code", true)).slice(0, 5);
  const reasoningRank = (await getRankedCandidates("reasoning", true)).slice(0, 5);

  return c.json({
    engine: "vajra-empirical-sovereign-v5.2-oct2026-verified",
    runtime: "deno-deploy-kv-cas-l1",
    security: {
      auth_enabled: Boolean(Deno.env.get("ROUTER_API_KEY")),
      admin_guard_enabled: Boolean(Deno.env.get("ADMIN_SECRET")),
      openrouter_free_enabled: Boolean(pickOpenRouterKey()),
      l1_cache_entries: L1_CACHE.size,
    },
    current_best_fast: fastRank,
    current_best_smart: smartRank,
    current_best_code: codeRank,
    current_best_reasoning: reasoningRank,
  });
});

honoApp.get("/v1/models", async (c) => {
  const ranked = await getRankedCandidates("smart", true);
  const virtualAliases = [
    "free:auto",
    "free:fast",
    "free:smart",
    "free:code",
    "free:reasoning",
  ].map((id) => ({
    id,
    object: "model",
    created: 1761523200,
    owned_by: "vajra-mesh-router",
  }));

  const discoveredModels = ranked.map((m) => ({
    id: m.modelId,
    tuple_id: m.tupleId,
    object: "model",
    created: 1761523200,
    owned_by: m.providerId,
    iq_effective: m.iqEffective,
    supports_reasoning: m.supportsReasoning,
    supports_vision: m.supportsVision,
    is_coding_specialist: m.isCodingSpecialist,
    context_window: m.contextWindow,
  }));

  return c.json({
    object: "list",
    data: [...virtualAliases, ...discoveredModels],
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
    const reqModel = String(body.model || "free:auto");
    const reqModelLower = reqModel.toLowerCase();
    const wantsStream = Boolean(body.stream);
    const hasTools = Array.isArray(body.tools) && body.tools.length > 0;
    const { hasVisionPayload } = extractMessageText(messages);

    const completionOptions: CompletionOptions = {
      temperature: typeof body.temperature === "number" ? body.temperature : undefined,
      max_tokens: typeof body.max_tokens === "number"
        ? body.max_tokens
        : typeof body.max_completion_tokens === "number"
        ? body.max_completion_tokens
        : undefined,
      top_p: typeof body.top_p === "number" ? body.top_p : undefined,
      reasoning_effort: typeof body.reasoning_effort === "string" ? body.reasoning_effort : undefined,
      tools: hasTools ? body.tools : undefined,
      tool_choice: body.tool_choice,
      response_format: body.response_format,
    };

    const mode: RoutingMode = reqModelLower.includes("auto")
      ? classifyPromptIntent(messages, hasTools, completionOptions.reasoning_effort)
      : reqModelLower.includes("fast")
      ? "fast"
      : reqModelLower.includes("code")
      ? "code"
      : reqModelLower.includes("reason") || reqModelLower.includes("think")
      ? "reasoning"
      : "smart";

    let ranked = await getRankedCandidates(mode, false, hasVisionPayload);
    if (ranked.length === 0 && hasVisionPayload) {
      ranked = await getRankedCandidates(mode, false, false);
    }
    if (ranked.length === 0) {
      return c.json({ error: { message: "No healthy discovered models available." } }, 503);
    }

    // If caller requested an explicit model ID (e.g., "nvidia/nemotron-3-ultra-550b-a55b:free"), prioritize matching tuples
    if (!reqModelLower.startsWith("free:")) {
      const exactMatches = ranked.filter(
        (cand) => cand.modelId.toLowerCase() === reqModelLower || cand.tupleId.toLowerCase() === reqModelLower,
      );
      if (exactMatches.length > 0) {
        const others = ranked.filter((cand) => !exactMatches.includes(cand));
        ranked = [...exactMatches, ...others];
      }
    }

    const timeoutMs = mode === "reasoning" ? 28000 : 18000;
    const masterAbort = new AbortController();
    const globalTimeout = setTimeout(() => masterAbort.abort(), timeoutMs);

    const wave = ranked.slice(0, 3);

    const attempts = wave.map(async (cand, idx) => {
      const staggerMs = idx === 0
        ? 0
        : idx === 1
        ? 250
        : mode === "fast"
        ? 1300
        : mode === "reasoning"
        ? 3000
        : 2200;
      if (staggerMs > 0) {
        await new Promise<void>((resolve, reject) => {
          const t = setTimeout(resolve, staggerMs);
          masterAbort.signal.addEventListener(
            "abort",
            () => {
              clearTimeout(t);
              reject(new Error("ABORTED_BY_WINNER"));
            },
            { once: true },
          );
        });
      }

      const provider = PROVIDERS.find((p) => p.id === cand.providerId);
      if (!provider || !provider.isConfigured()) throw new Error("Provider missing or unconfigured");

      try {
        const res = await provider.executeCompletion(
          cand.modelId,
          messages,
          masterAbort.signal,
          completionOptions,
        );
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
      const finishReason = winner.toolCalls && winner.toolCalls.length > 0 ? "tool_calls" : "stop";

      if (wantsStream) {
        const stream = new ReadableStream({
          start(controller) {
            if (winner.reasoningContent) {
              const reasoningChunk = {
                id: completionId,
                object: "chat.completion.chunk",
                created,
                model: winner.candidate.modelId,
                provider: winner.candidate.providerId,
                choices: [
                  {
                    index: 0,
                    delta: { role: "assistant", reasoning_content: winner.reasoningContent },
                    finish_reason: null,
                  },
                ],
              };
              controller.enqueue(encoder.encode(`data: ${JSON.stringify(reasoningChunk)}\n\n`));
            }

            const chunkPayload = {
              id: completionId,
              object: "chat.completion.chunk",
              created,
              model: winner.candidate.modelId,
              provider: winner.candidate.providerId,
              choices: [
                {
                  index: 0,
                  delta: {
                    role: "assistant",
                    content: winner.content,
                    ...(winner.toolCalls ? { tool_calls: winner.toolCalls } : {}),
                  },
                  finish_reason: null,
                },
              ],
            };
            const stopPayload = {
              id: completionId,
              object: "chat.completion.chunk",
              created,
              model: winner.candidate.modelId,
              choices: [{ index: 0, delta: {}, finish_reason: finishReason }],
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
          supports_reasoning: winner.candidate.supportsReasoning,
          is_coding_specialist: winner.candidate.isCodingSpecialist,
          reliability_draw: winner.candidate.reliabilityDraw,
          observed_ttft_ms: winner.ttftMs,
          observed_total_ms: winner.totalMs,
          ewma_tps: winner.candidate.ewmaTps,
        },
        choices: [
          {
            index: 0,
            message: {
              role: "assistant",
              content: winner.content,
              ...(winner.reasoningContent ? { reasoning_content: winner.reasoningContent } : {}),
              ...(winner.toolCalls ? { tool_calls: winner.toolCalls } : {}),
            },
            finish_reason: finishReason,
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

if (import.meta.main) {
  Deno.serve(app.fetch);
}

export { app };
export default {
  fetch: app.fetch,
};
