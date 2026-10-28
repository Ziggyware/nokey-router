# Nokey Router update — 2026-10-27

Base: GitHub main @ 9aa0877 (the services.json build). Deploy = redeploy the
Deno project with this main.ts; a push alone does not deploy.

## Why text generation was 502ing on production
POST /v1/chat/completions existed and the keyless providers are alive (Kilo
and Pollinations both answered direct test calls), but every lane returned
502 "All promises were rejected". Two dispatch defects:

1. The dispatch race and the canary probe resolved providers with
   PROVIDERS.find ONLY, ignoring the services.json registry providers that
   the ranking admits. Registry candidates (ids services_json_*) instantly
   threw "Provider missing or unconfigured". Since providers moved into
   services.json — and the native Pollinations spec was removed in 9aa0877 —
   most of the catalog could rank but never execute.
2. Waves draw 3-at-a-time from the head of the ranking (9 attempts max) with
   no provider diversity, so one failing provider's tuples (production's
   head was all OpenRouter) burn every attempt while healthy providers are
   never tried.

## Fixes in main.ts
- resolveProviderById(): merged lookup across native PROVIDERS and the
  services.json registry; used by the dispatch race and the canary probe.
  This also makes the 9aa0877 Pollinations removal safe: Pollinations lives
  on in services.json (5 endpoints) and is dispatchable again.
- interleaveByProvider(): ranked lists are round-robined across providers
  before racing, per-provider order preserved.
- Provider circuit breaker: 6 consecutive tuple failures exclude a provider
  from ranking for 5 minutes; its next success clears it.

## New: image generation (OpenAI-standard path)
POST /v1/images/generations  { model, prompt, size, n, response_format }
-> { created, data: [{ url }] } or [{ b64_json }] with response_format=b64_json.
Models (also listed by GET /v1/models):
- nokey-image       Pollinations turbo (fast) — primary
- nokey-image-flux  Pollinations flux (quality, slower)
- nokey-image-horde AI Horde stable_diffusion (anonymous key)
Pollinations failures fall back to the Horde automatically.
NOTE: the image endpoint calls the Pollinations image API directly; it does
not depend on the removed native text spec.

## Page (index.html, new in repo)
Dashboard media tab POSTs {router}/v1/images/generations with the header-bar
API key, shows a spinner, renders data[0].url, sets the download link, logs
to the Inspector tab. Video option marked unavailable (no keyless provider).

## Verified locally against this exact build (with services.json present)
- deno check clean
- GET /v1/models lists nokey-image, nokey-image-flux, nokey-image-horde
- POST /v1/chat/completions with a realistic request -> 200 in ~3s (Kilo);
  canary probe passes; registry loads (334 catalog entries) and registry
  tuples appear in rankings again
- POST /v1/images/generations returns real images via both backends
  (Pollinations 503s are absorbed by the Horde fallback)

## Known wart (pre-existing, not fixed here)
Tiny max_tokens probes (e.g. 16) against reasoning-heavy models return
502 after long wave-burning: the model spends the whole budget on
reasoning, content arrives empty, and verifyCleanText throws POISON_EMPTY.
Realistic budgets (the dashboard/Forge defaults) are unaffected. Worth a
follow-up: treat POISON_EMPTY as a soft failure that returns the truncated
completion instead of failing the candidate.

## Production note
The live deployment failed even a realistic request in ~5.6s (every
candidate rejected almost immediately) while this build answers the same
request locally in ~3s — the deployed code itself is the broken layer.
Redeploying this build is the fix; if it still 502s after redeploy, check
the deployment's OPENROUTER_API_KEY next.

## Addendum — 2026-10-27 (status moved to /health)
2822769 made the router serve this dashboard at GET / itself and moved the
JSON gateway status to GET /health. Every consumer still parsing / as JSON
(the dashboard Sync check, the connector status helper) died with an
"HTML is not valid JSON" parse error.
- index.html: connection test now probes GET /health; a fetchJson helper in
  the Architect/media paths and an HTML guard on the chat stream turn the
  raw parse error into a plain-English message.
- Connector helper (router.py status): falls back to /health when / does
  not return JSON, instead of crashing with a decode traceback.
- main.ts: unchanged by this addendum (the /health route already exists).

## Addendum 2 — 2026-10-27 (unverified tuples no longer top the IQ lanes)
After the redeploy, free:fast and image generation worked on production,
but free:smart / free:code / free:reasoning still 502ed: their rankings are
IQ-sorted, and newly discovered tuples entered the catalog with
iq_verified = 1.0 (full trust) plus an optimistic 900 ms latency prior.
Registry gateways advertising frontier names ("claude-opus-5",
"gemini-4-argon") therefore monopolized the IQ lanes despite never having
served a completion, and every wave burned on them.
- Fresh tuples now start unverified: iq_verified 0.55 (the canary's fail
  value) and a neutral 2500 ms latency prior.
- Sync clamps iq_verified to 0.55 for any tuple that has never succeeded,
  so existing catalogs self-heal without a KV wipe.
- Each real dispatch success raises iq_verified by 0.15 (cap 1.0); the
  canary still sets 1.0 / 0.55 outright on probe results.
Verified locally: after a sync, Kilo natives head the smart/code rankings
and both lanes complete in seconds (previously 502).

## Addendum 3 — 2026-10-27 (rank-time trust clamp)
Addendum 2's demotion ran during catalog sync, but sync skips providers
whose model discovery fails -- the exact population that needed demoting --
so production's IQ lanes stayed clogged after the redeploy (fast and code
recovered; smart/auto/reasoning still 502). The clamp now also applies at
rank time in getRankedCandidates: any tuple with success_count == 0 is
scored with iq_verified capped at 0.55 on every request, regardless of the
stored value or sync state.

## Addendum 4 — 2026-10-27 (rolling hedged race + dashboard overhaul)
Server: the wave race (3 candidates per wave behind 0/250/1300-3000 ms
staggers and a wave barrier) is replaced by a rolling hedged race: first
candidate starts immediately, hedges join at +150/+400 ms, and EVERY
failure instantly launches the next candidate (up to 12 attempts, max 5
in flight) -- a fast 4XX no longer leaves the next attempt unstarted for
~1 s. Catalog sync no longer blocks the request path: with a warm cache
a due sync refreshes in the background (cold start / admin force still
await). Local proof: fast 1.9 s, smart 2.2 s, and an explicit request for
a dead registry model fell through its broken tuples to a working
provider in 2.7 s (previously 7-10 s of waves, usually a 502).
Dashboard: the header and tab nav are merged into one unified top bar
inside a fixed-height app shell (h-screen flex column) -- the bar never
scrolls, and the chat composer is pinned: only the message list scrolls.
Self-hiding panels: the connection bar and the chat Routing Parameters
panel collapse via toggles (state persists; params auto-hide on small
screens). New Plugins tab: a real registry (NokeyPlugins.register) with
four working built-ins (Lane Probe, Catalog Sync, Model Ping, Image
QuickGen) plus an Install panel that mounts Architect-generated plugin
code live. New Admin tab: locked until the existing Admin Secret is
verified against the router with a live probe; then status cards, lane
rankings, probe runner, and catalog force-sync.

## Addendum 5 — 2026-10-27 (frontier resilience: escalate, resurrect, salvage)
Three changes so the router expands its choices instead of giving up:
1. Second-wind escalation. When the tier-1 race (12 attempts) exhausts,
   dispatch escalates instead of returning 502: tier 2 widens to the next
   candidates INCLUDING providers in circuit-breaker cooldown (mercy
   attempts; breaker state is a stale snapshot), and tier 3 rescues
   cross-lane to any tuple that has ever completed a request (this lane,
   then the fast lane). getRankedCandidates gained includeCoolingDown /
   verifiedOnly options; every attempt is tracked so tiers never retry a
   tuple. Tier-1 size is env-tunable (MESH_RACE_TIER1_ATTEMPTS). Proven
   locally with tier-1 forced to 1 attempt on a proven-dead tuple: the
   log shows tier 2 engaging (273 candidates) and rescuing the request
   in ~2.2 s.
2. Long-tail resurrection. Of ~1,300 catalog tuples most sit unproven
   (success_count == 0) and the canary only re-tests the top 4. A new
   prober (cron every 10 min, POST /v1/resurrect, Admin tab button)
   walks the least-recently-probed tail with the canary question and
   promotes whatever answers -- verified IQ written back exactly like
   the canary. First local run: probed 4, promoted 1
   (kilo::cohere/north-mini-code:free), correctly buried 3 dead tuples.
3. Salvage, don't discard. A reasoning-only completion (empty visible
   content, full reasoning_content) used to die as POISON_EMPTY. The
   parser now promotes the reasoning into content -- the same promotion
   splitThinkBlock already did for <think> blocks -- and counts it as a
   win. Tool-call responses were already exempt; truly empty responses
   still throw. (Branch is type-checked and mirrors the existing think-
   block behavior; no live reasoning-only response occurred during
   local testing despite several tiny-token attempts.)

## Addendum 6 — 2026-10-27 (the Architect gets hands: tool-use agent)
The Vajra Core Architect was a chat box: one lane call, no abilities. It
is now a real tool-use agent in the page. Protocol: the model replies
with a fenced ```action {"tool": ...}``` block; the page executes the
tool, feeds the result back, and loops up to 6 rounds, rendering each
tool call in the chat. Tools: read_file (the system's real source —
main.ts, index.html, CHANGES.md, services.json, deno.json — paged 300
lines per call), router_health, list_models, list_plugins,
install_plugin (mounts live into the Plugins registry), ping_model,
run_probe and force_sync (admin). New server endpoint GET /v1/source
(admin-guarded, strict allowlist, path traversal rejected) serves the
source files; without the Admin Secret the Architect can still read
index.html (its own page) and use the non-admin tools, and the tab shows
whether admin tools are armed. The system prompt grounds it honestly:
deployments are immutable, plugins are its writable surface, and server
changes must be handed over as patch text, never faked. Verified three
ways: the page's own tool code ran in a Node harness against the live
local server (12/12 checks: parsing, health, source reads, allowlist,
install/list plugins, ping, unknown-tool handling); a live model
followed the action format, consumed a real list_models result and
answered with the true catalog count (277); and it produced a
well-formed install_plugin action for a clock plugin on request.

## Addendum 7 — 2026-10-27 (Architect v2: byte budgets, code navigation, plugin SDK v2)
Fixes from live use. (1) /v1/probe returned raw Promise.allSettled
entries while every consumer (Admin tab, Lane Probe plugin, Architect)
expected {tupleId, ok, passed, ms} — probes misreported as failures.
The endpoint now normalizes results and probe failures carry their
tupleId; all three consumers render PASS / ANSWERED (wrong canary
answer) / FAIL distinctly. (2) read_file was line-capped (300 lines,
14k chars). Reads are now BYTE-budgeted: default 120,000 chars per call
(up to 200,000), whole lines from start_line, with the next start_line
reported — the whole 83 KB / 2,216-line main.ts returns in ONE call.
Source fetches are cached per session. (3) Two navigation tools join
read_file: search_code (regex over a file, line-numbered matches) and
file_map (structural outline — top-level functions/consts, routes,
crons, sections, headings; local-variable noise filtered after testing
showed it drowning the map). (4) The action parser now accepts ```json
fences, bare JSON objects, and trailing commas, and if a reply looks
like an action but does not parse, the loop feeds back a correction and
lets the model resend instead of ending the turn; rounds raised 6 -> 8.
(5) Plugin SDK v2: ctx gains chat, chatStream (SSE), image, models,
health, probe, sync, resurrect, source, and storeFor (persistent
per-plugin storage). Model Ping and Image QuickGen were rebuilt on it,
and a new built-in, Prompt Lab, is a streaming chat app with a model
picker and persisted history that exists purely on SDK calls.
Verified: harness v2 against the live local server — parser variants
4/4, byte-budgeted whole-file read, search, file map, ctx.chat /
ctx.models / ctx.storeFor / ctx.probe all passing; a live model asked
to locate the resurrection probe chose file_map unprompted.
