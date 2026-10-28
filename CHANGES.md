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
