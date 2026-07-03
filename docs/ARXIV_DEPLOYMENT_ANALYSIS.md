# arXiv-scale deployment bottleneck analysis

**Date:** 2026-07-03. **Scope:** what breaks when `bibtex-check` is deployed to screen all incoming arXiv submissions for hallucinated references — tradeoffs, missing infrastructure, latency, orchestration, and the current pipeline.

**Method:** 25 read-only analysis agents in two workflows: six parallel deep dives (pipeline, latency, cache/state, I/O boundary, accuracy artifacts, external landscape) → capacity model → 10 claims adversarially verified → completeness critic → cost and gap probes → 4 patch probes, including an **empirical measurement** of cascade load from the eval-run HTTP caches. Key numbers below are measured or verified against `file:line`/artifacts, not estimated. Line numbers refer to the state of `main` at commit `d6ea05d`.

## Context

arXiv ≈ 933 submissions/day (~27.5–28k/month, confirmed vs arxiv.org/stats) × ~45 refs/paper → **~42k references/day (~1.2M/month)**, arriving as LaTeX source (usually a compiled `.bbl`, never `.bib`), feeding a moderation flow with an hours-to-a-day budget (AutoTeX retired summer 2025 — no synchronous compile-path hook).

**Headline:** the checker's accuracy is not the deployment problem. The blockers are (1) a hard input-format gap, (2) precision collapse at population base rates that makes *human review labor* ~97% of total cost, and (3) fail-open verdict semantics that silently lose recall exactly when the system is under load. The live-API architecture is marginal at 1× and cannot scale horizontally; the viable shape is snapshot-first + live fallback for recent refs, emitting ranked triage (never auto-reject), with per-domain calibration.

## Current pipeline (verified)

- **Cascade (5 members, strictly sequential per entry):** CrossRef → (S2 `/paper/search/match` early, key only) → OpenAlex → DBLP → OpenReview → S2 relevance search (skipped if the early match ran) — `fact_checker.py:3409–3573`. arXiv is a pre-cascade consistency check/fast-path + post-cascade by-ID candidate; OpenLibrary/GoogleBooks are a separate book path; EuropePMC/ACL live in the resolver only.
- **Short-circuit** = `_has_full_confirmation`: score ≥ 0.95 **and** every claimed field confirmed (`:3695–3708`) — a 0.99 title match that can't confirm the claimed venue keeps traversing. One search call per source (top-k=3 re-ranked locally, no per-candidate fetches); OpenReview is the exception at ~2.9 req/query (search + per-note fetches). A relaxed-author retry adds 2 calls when the strict pass finds nothing — i.e., **hallucinated entries (the screening target) are systematically the most expensive** (worst case ~8–10 req/entry).
- **Concurrency:** ThreadPoolExecutor, 8 workers default; two 16-thread prefetch pools (Crossref record warm + DOI pre-validation); httpx pool 50/20, 20s timeout, 6 retries with 1→16s backoff + Retry-After; adaptive per-service rate-halving on 429.
- **CLI rate limits** (`_cli_service_rate_limits`, `:5580–5609`, verified directly): crossref 300/min (cap 600), openalex 150/min (cap 300), dblp/openreview 30 (cap 60), **arxiv 20 flat (never scaled)**, S2 **60/min keyed = exactly 1 req/s ToS** / 10 keyless. The library `DEFAULT_LIMITS` (crossref 50 etc.) are dormant defaults the CLI overrides.
- **One-shot CLI, no service layer:** no server/health/queue/signals/metrics; plain-text logging; `X-From-Cache` set but never read; rate-limit headroom never exported. `build_checker_processor`/`process_entries` are the embedding hooks (`:5882–5987`).

## Measured load (replaces all prior estimates)

From the run-isolated SQLite caches of the two HALLMARK eval runs (timestamps match the run windows; 3,926 unique HTTP requests / 1,950 entries):

- **2.01 unique requests/ref** (dev 2.19, test 1.78; S2 keyless → 0 S2 calls). Docs claim ~1.4 (optimistic). With keyed S2 in production: **~2.6–3.2 req/ref**.
- At 42k refs/day: **~84k live calls/day** (S2 off) → **~115–135k/day** (S2 on).
- Request share: Crossref 39% (mostly DOI-direct `/works/{doi}`), **OpenReview 33%** (the hidden consumer via per-note fetches), OpenAlex 20.5%, DBLP 3.8%, arXiv 3.6%.
- **Latency is throttling-dominated, bimodal:** dev ran ~3.4s/entry effective; test collapsed to 611–723s/batch (~25–29s/entry, 7–20× slower) under sustained throttling, dropping 20 entries. At ~3.4s/entry, 42k refs/day ≈ 1.6 process-equivalents 24/7 — feasible; in the throttled regime it needs ~12–14, and each added process multiplies uncoordinated outbound (per-process limiter) → bans. **The instability, not the mean, is the risk.**

Per-provider fit at 1× (84k/day): Crossref ~33k ✓ (polite 10/s single-record); OpenReview ~28k ✓ (permissive ToS); **OpenAlex ~17k searches/day = ~170k credits vs 100k free credit cap ✗** (keys now required; search billed $1/1k → ~$510/mo, or snapshot); DBLP ~3.2k ✓ (monthly dump available anyway); arXiv ~3k vs 28.8k flat ✓; S2 keyed 42k vs 86.4k ceiling ✓ but consumes half.

## Bottleneck ranking

### 1. Input gap — no `.bbl`/LaTeX ingestion (hard blocker)
`.bib`-only via bibtexparser (`:6041–6046`, CONFIRMED; no GROBID/refextract anywhere). arXiv ships LaTeX source whose bibliography is usually a compiled `.bbl` with field structure destroyed. Worse: **a malformed file aborts wholesale (exit 1)**, and sparse entries (title+author only) silently disable the year/venue/DOI/arXiv detectors — so extraction *quality*, not just existence, gates detector coverage. Parse `.bbl`/`thebibliography` from arXiv S3 source (near-structured, sidesteps GROBID's 10–20% error noise that would masquerade as hallucinations); PDF fallback only.

### 2. Precision collapse at population base rate (statistical blocker)
Operating point (shipped build, verified): dev DR 0.848 / FPR 0.0468; test 0.884 / 0.0417. Precision-of-flag = b·d/[b·d+(1−b)·f] reaches 50% only at base rate **b\* = 5.2%**; measured fabrication rates are ~1% of *papers* (≪1% of refs). Consequences:
- **False-flag floor ≈ 42k × 0.0468 ≈ 1,960/day regardless of base rate**; at b=0.1% precision ≈ 1.8% (49 of 50 flags false).
- **Off-ML-domain FPR = 32%** (PubMed split; venue + given-name misparses) — a single global threshold false-flags ~1 in 3 valid non-ML refs; domain mix swings labor 3–7×.
- **Cost (probed):** review at 5 min/flag ≈ **$112–374k/mo (center ~$185k/mo) — >97% of total cost**; the entire API/infra question ($0.7–7.5k/mo) is noise. The only real lever is capping review to a ranked top slice (~$26–93k/mo), trading recall on unreviewed flags.
- The triage escape hatch requires recalibration first: **p_valid ECE 0.21–0.28** — thresholds don't transfer across splits/domains.

### 3. Fail-open semantics under load (silent recall loss)
`api_error` → VALID unconditionally; `not_found` + `coverage_incomplete` → VALID (`eval_hallmark.py:69,346–348`; `fact_checker.py:302–331`). ~16–21% of entries abstain; **87 of 92 dev misses are abstention pass-throughs**, concentrated in exactly the DOI-less shapes LLMs hallucinate (wrong/nonexistent venue, partial author lists, preprint-as-published — detection 0.44–0.65). So throttling, circuit-breaks, or snapshot staleness **convert would-be flags into silent passes** — and the measured eval already hit this (throttled entries re-run; HALLMARK harness even backfills timeout-missing keys as VALID). Production recall < offline 0.85 whenever the API budget strains; an adversary can induce the strain. Blind spot ≡ attack surface.

### 4. Global provider ceilings vs per-process rate limiting (major)
`RateLimiter` is in-memory per process; providers enforce per key/IP. 1× is marginal (OpenAlex credits already exceeded); horizontal scaling multiplies real outbound with zero coordination. `_doi_resolves` runs on a **raw httpx client bypassing limiter and cache** (`:1852–1878`, ~42k uncoordinated doi.org probes/day); several verifiers hardcode a no-mailto UA. Every provider steers this volume to bulk dumps.

### 5. Cache wastes the workload's structure (major)
Keys = full HTTP request signature; caches only 200s; **no negative, DOI-probe, or verdict caching** (every re-run recomputes everything); 30-day lazy TTL, `clear_expired` never called, unbounded growth; WAL cannot cross hosts (M machines = M cold caches = M× provider load); circuit-breaker state loaded once at boot, last-writer-wins. The workload's biggest gift — the same popular papers cited across thousands of submissions — is only partially captured.

### 6. No service layer (major, mechanical)
Queue consumer, health, SIGTERM/WAL-checkpoint, structured logs, per-service counters, rate-headroom gauge, cache hit ratio — all absent. JSONL streams append-mode per entry (partial results survive a kill) but **no resume** (re-runs duplicate lines). Non-strict exit code is always 0 — a gate must parse JSONL; `--strict` is the intended gate mode. JSONL provenance is thin: `mismatched_fields` is names-only, no per-field source attribution — weak for "why was this flagged" moderation UX.

### 7. Unaudited pass-through paths (adversarial, cross-cutting)
Once public, authors adapt: (a) chain.py trusts (never fact-checks) upgrades passing title ≥0.90 ∧ author ≥0.80 ∧ year ≤+5 — a fabrication fuzzy-matching a real preprint record passes silently, and no artifact measures the vouched fraction; (b) DOI-less soft fabrications abstain→VALID (#3); (c) non-Latin/transliterated author names are deliberately softened to could-not-verify (`:4504–4510`) — a systematic multilingual PASS path with no benchmark split; (d) **retraction blindness (probed):** zero code reads retraction metadata; `PublishedRecord` has no field for it; a real-but-retracted paper is *actively certified* VERIFIED. Fix is **parse-only** — OpenAlex `is_retracted` and Crossref `update-to` already arrive in fetched responses (no field projection is used); Retraction Watch is now Crossref-owned CC0/CC-BY. Cross-check both (OpenAlex had a 2023-12→2024-03 false-positive window, arXiv:2403.13339).

## Missing infrastructure (ranked)

1. **`.bbl`/LaTeX reference extractor** (blocker; medium) — `.bbl`/`thebibliography` parser over arXiv S3 source; refextract/GROBID PDF fallback; field-recovery quality gates detector coverage (see #1).
2. **Calibration + domain-aware thresholds + capped-review triage loop** (blocker mitigation; medium) — per-domain operating points, p_valid recalibration, ranked review queue, feedback capture.
3. **Local snapshot mirror + entity resolution** (high; 300GB–1TB, ETL, ~$0.7–3k/mo) — OpenAlex + Crossref + DBLP (+S2 datasets); removes #4 for ~90% of refs. **Critical freshness rule:** refs with claimed year ≥ snapshot month must route to live API, else stale-miss + fail-open (#3) silently passes recently-fabricated refs.
4. **Service layer** (medium) — queue consumer around `build_checker_processor`, **global distributed rate coordinator**, health/SIGTERM/WAL-checkpoint, resumable output.
5. **Shared semantic cache** (medium-high) — Redis/Postgres keyed on canonical IDs; verdict + negative + DOI-probe caching; shared circuit state.
6. **Retraction check** (low; parse-only) + **observability** (medium; counters, headroom gauge, structured logs, per-field provenance in JSONL).

## Orchestration tradeoffs

- **On-submit streaming:** 39 subs/hr ≈ 0.38 refs/s — trivial when warm and well-behaved; but requires the whole missing service layer, and the throttled regime (7–20× slowdown) makes latency SLOs fragile.
- **Nightly batch:** maximizes cross-submission dedup, fits the moderation window; bursts provider load into a window (worsens the throttling feedback loop that triggers fail-open passes).
- **Hybrid (recommended):** snapshot-first, live fallback gated on recency; async/advisory in the moderation window; ranked triage with capped human review; per-domain thresholds; strict mode. The only shape that meets 933/day within ToS *and* keeps the fail-open path cold.

## Verification verdicts (adversarial pass)

CONFIRMED: per-process limiter / no global coordination; WAL same-host; `.bib`-only; dev FPR 0.0468/DR 0.848 + false-flag floor + b\*=5.2%; 32% cross-domain FPR; fail-open mapping. REFUTED: "S2 429s at any volume" (CLI caps S2 at exactly 1 req/s keyed). PARTIAL: SQLITE_BUSY severity (Python's default 5s busy handler; sustained contention still unhandled); OpenAlex cap is a *credit* budget (search = 10 credits); load ≈ 42k not 40k refs/day.

**Repo-facing documentation notes surfaced along the way:** `docs/REFERENCE_FACT_CHECKER.md:227` claims ~1.4 API calls/entry (measured: 2.01); `benchmarks/HALLMARK.md`'s header table still shows v1.2.0 numbers while `eval_runs/` carries the current build's results; the installed version string (`1.3.1.dev18`), CHANGELOG (`[1.4.0]`), and benchmark label ("unreleased c758f7e") disagree on what "shipped" means — an auditability gap for a policy-enforcing tool. No AUROC metric is computed anywhere despite occasionally being quoted.

## Possible follow-up work (each independently shippable)

- **P0 (unblock):** `.bbl`→BibTeX extractor + per-entry (not per-file) parse robustness; retraction parse (OpenAlex `is_retracted` + Crossref `update-to` → new `RETRACTED` status behind a flag).
- **P1 (make it honest under load):** distinguish "checked-and-clean" from "couldn't check" in the output contract (stop folding abstentions into VALID for the gate consumer); export rate-headroom/coverage metrics so throttling-induced recall loss is visible.
- **P2 (make it scale):** snapshot mirror + recency-routed live fallback; shared semantic cache; global rate coordinator; queue/service wrapper.
- **P3 (make it deployable):** per-domain calibration + capped-review triage UI contract (per-field provenance in JSONL).

## Reproducing the measurements

1. Cascade load: the 2.01 req/ref figure came from `eval_runs/*/.cache.fact_checker.db` unique-URL counts vs entry counts — rerun the query after any pipeline change.
2. Timed end-to-end: `bibtex-check` on a 45-ref bib, cold vs warm `--cache-file`, watch limiter waits with `--verbose`; compare against the 3.4s vs 25–29s/entry regimes in `eval_runs/*/eval.log`.
3. `.bbl` pilot: pull ~100 arXiv source tarballs, naive `\bibitem` parse, measure structured-field recovery (bounds blocker #1 and the detector-dormancy risk).
4. Domain-mix sensitivity: HALLMARK crossdomain split with per-domain thresholds to bound the labor multiplier (3–7×).
