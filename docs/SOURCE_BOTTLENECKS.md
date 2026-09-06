# Source bottlenecks in citation verification

What each source does under load, what it is allowed to do, what each repo does about it,
and what to set before a run.

## Summary

dblp is sick server-side: it answers `503` with the body "No server is available to handle
this request." and no `Retry-After`, an empty backend pool rather than throttling
(SOURCE_HEALTH.md, 2026-09-04). Semantic Scholar is broken by a credential the AWS gateway
refuses exactly as it refuses an invented key, 0 of 18 probes on 2026-09-04. Four were
overdriven by us rather than failing: arXiv (three shards each taking the full per-caller
budget), CrossRef (1,800/min offered against a 180 allowance), OpenReview (one login per
host per thread), and dblp under the same sharding. OpenAlex is healthy with a key and
bound by credits, not rate. Google Scholar blocks harcx, and doi.org's redirect targets
answer `202` or `403` from publisher bot mitigation, which three call sites read as a
missing DOI.

## dblp

**Observed.** On 2026-09-04 dblp answered 10 of 15 probes at one request per 45 seconds
and 30 of 48 (62%) at one per 8 seconds; the five-minute probe log gave 81 of 227 (36%):
92 empty responses, 37 × 503, 16 × 500 and exactly one 429 (SOURCE_HEALTH.md). Across all
501 samples in `screening/source_availability.log` (file last written 2026-09-05 11:08; it
stamps times, not dates) dblp answers 201 times, 40.1%, with 2 × 429 among the 300
failures. Availability tracks the clock, not our load: 0–27% between 12:00 and 17:00 local
against 42–75% between 00:00 and 04:00, with failures no more frequent at one request per
eight seconds than at one per five minutes. In the HALLMARK pre-screening ablation of
2026-09-04/05, dblp is 86% to 95% of every incomplete-lookup column: 275 of 285, 230 of
244, 205 of 223, 155 of 163 (source-availability-is-a-measurement-condition.md). A single
probe on 2026-09-06 at 07:55 UTC (a Sunday morning) answered in 6.25 s while every other
source answered under 0.6 s; Semantic Scholar keyless returned 429.

**Documented limit.** None published. The crawling FAQ asks for 1–2 s between requests and
says throttling arrives as `429` with `Retry-After` (https://dblp.org/faq/1474706.html),
which one 429 in 227 samples rules out.

**bibtexupdater.** 1.8.0 demotes an exhaustive miss to `api_error` whenever any source
failed, records `sources_failed` per entry, and exits 5 once the affected fraction reaches
`--outage-threshold` (default 10%). 1.8.1 orders sources by health from the existing
circuit breaker and doubles a reopened circuit's cooldown from 90 s to a 30-minute cap,
against a probe that found dblp unreachable in 32 of 36 samples across two days (CHANGELOG
1.8.1).

**hallmark.** `hallmark/baselines/bibtexupdater.py`'s `_run_bibtex_check_subprocess` raises
`SourceOutageError` on exit 5 rather than scoring the run, `parse_source_condition` captures
the per-source failure counts, and `hallmark/cli.py`'s `_stamp_provenance` stamps them onto
`EvaluationResult.source_condition`.

**Recommended and open.** 30/min aggregate (10/min per shard at N=3); treat a miss as
unknown, never as not-found. The cause is unknown and no status page was found
(SOURCE_HEALTH.md, "Unverified"); the 2026-09-03 ruling keeps dblp in InterpScience's
ordering, where it gates no verdict (STATE.md).

## arXiv

**Observed.** 15 of 15 baseline probes on 2026-09-04, 48 requests up to 60/min with
latency flat near 95 ms, and 18 of 18 at three concurrent connections × 20/min. Under the
real run it starved: three shards each took the stock 20/min, offering 60/min; arXiv
answered 429, the circuit opened on the fourth failure with the cooldown escalating to
1800 s, and arXiv covered 90 of 750 lookups (12%) while a single-process probe on the same
entries reached it 6 of 6 (STATE.md, 2026-09-03). After `BIBTEX_ARXIV_RATE=6`, 18 of 18
under the same load. A second failure shape outlasts pacing: on 2026-09-04 the same 20/min
aggregate was refused whether delivered on one connection or three (see the throughput
section), a cooldown from earlier overload rather than a rate to tune to (STATE.md).
Starvation changes verdicts: on a 119-entry probe with model and prompt fixed, 12% of
flags cleared with arXiv starved against 24% with it answering
(source-availability-is-a-measurement-condition.md).

**Documented limit.** 1 request per 3 s, single connection at a time
(https://info.arxiv.org/help/api/tou.html).

**bibtexupdater.** 1.10.3 keeps arXiv at a flat 20/min that `--rate-limit` never scales,
and `BIBTEX_ARXIV_RATE` lets a sharding launcher divide the caller budget
(`src/bibtex_updater/fact_checker.py`'s `_cli_service_rate_limits`).

**hallmark.** Single-process, so the 20/min default is the whole budget.

**Recommended and open.** 20/min per caller across every process, `BIBTEX_ARXIV_RATE=6` at
N=3. The knee sits above 60/min and was not found, and the sustained-access cooldown is
uncharacterized beyond outlasting a 150 s test (SOURCE_HEALTH.md, "Unverified").

## CrossRef

**Observed.** 15 of 15 baseline probes on 2026-09-04, median 0.40 s, then 48 consecutive
requests up to 60/min with no refusal. Live headers settle the pool question: with
`mailto`, `x-rate-limit-limit: 3` and `x-api-pool: polite-array`; without it, limit 1 and
`public-array`. The 501-sample probe log has 10 × 429 (97.6% answered).

**Documented limit.** Polite pool 3 req/s for list queries and 10 for single-record
lookups, public pool 1 req/s, effective 1 December 2025
(https://www.crossref.org/blog/announcing-changes-to-rest-api-rate-limits/).

**bibtexupdater.** CrossRef gets 300/min at scale 1.0, cap 600/min
(`src/bibtex_updater/fact_checker.py`'s `_cli_service_rate_limits`); its docstring still says
the polite pool advertises about 50 req/second, which the December 2025 change made wrong.

**hallmark.** The wrapper passes neither `--mailto` nor `--openalex-mailto`
(`hallmark/baselines/bibtexupdater.py`'s `_run_bibtex_check_subprocess`), so a run sits in
the public pool at 1 req/s while its default `--rate-limit 120` scales CrossRef to 600/min,
about 10 req/s. Exporting `BIBTEX_CHECK_MAILTO` closes that without a code change
(`src/bibtex_updater/fact_checker.py`'s `_resolve_polite_mailto`).

**Recommended and open.** 180/min aggregate (60/min per shard at N=3), contact address
always set. The ceiling was never reached at the 60/min test cap.

## OpenAlex

**Observed.** 15 of 15 keyed probes on 2026-09-04, median 0.76 s. The key reports
`x-ratelimit-limit: 10000` per day where the documentation quotes 100,000 for free users,
plus a one-time balance of 106,810 credits that does not refill. A `?search=` query costs
10 credits and a `/works/` DOI lookup costs 1, both read from `x-ratelimit-credits-used`,
so the daily budget buys 1,000 searches and a 5,043-reference pass exceeds it in one run
(SOURCE_HEALTH.md). A keyless call at 05:14 UTC returned 429 with `x-ratelimit-limit:
1000`, `remaining: 0` and a reset 67,376 s out while the key returned 200 in the same
minute: a keyless 429 is a spent daily budget, never an outage. HALLMARK ablation failures
were flat at 26, 26, 26 and 16 per arm, a credit ceiling rather than a rate.

**Documented limit.** 100,000 credits/day and 100 req/s
(https://raw.githubusercontent.com/ourresearch/openalex-docs/main/how-to-use-the-api/rate-limits-and-authentication.md).

**bibtexupdater.** 150/min at scale 1.0, cap 300/min
(`src/bibtex_updater/fact_checker.py`'s `_cli_service_rate_limits`); `--openalex-api-key` or
`OPENALEX_API_KEY` (`src/bibtex_updater/fact_checker.py`'s `build_checker_processor`)
bypasses the keyless budget.

**hallmark.** Passes no OpenAlex flag, so the key reaches the tool only through the
environment, and `scripts/check_source_reachability.py`'s `PROBES` table probes it keyless,
where a spent budget reads as "throttled, reachable" rather than the condition the run will
see.

**Recommended and open.** Rate is not the lever: 300/min aggregate is safe, and credit
cost comes down by routing through `/works/https://doi.org/...` wherever a DOI exists. Why
the key reports 10,000/day against a documented 100,000 is unresolved (SOURCE_HEALTH.md,
"Unverified").

## Semantic Scholar

**Observed.** The key returned 403 on all 18 probes on 2026-09-04. An authenticated call
returns `HTTP/2 403`, `x-amzn-errortype: ForbiddenException` and a 23-byte body; a key
invented on the spot returns the identical status, error type and byte count; no key at
all returns 429 with `TooManyRequestsException` and a 174-byte body. Both `/paper/search`
and `/paper/DOI:` behave the same way, so the gateway refuses before Semantic Scholar sees
it. A dead credential is worse than none: dropping the expired key took the
incomplete-lookup rate from 70% to 27.5% (eval-hardening-2026-09-04.md).

**Documented limit.** Keyless 1,000 req/s shared across every anonymous user; a key gives
1 req/s (https://www.semanticscholar.org/product/api).

**bibtexupdater.** 60/min with a key, otherwise `max(5, 10 × scale)`
(`src/bibtex_updater/fact_checker.py`'s `_cli_service_rate_limits`).

**hallmark.** The wrapper forwards `S2_API_KEY` as `--s2-api-key` whenever it is set
(`hallmark/baselines/bibtexupdater.py`'s `_run_bibtex_check_subprocess`), which is exactly
the 70% condition while the key is dead.
`run_shards.sh` unsets it for the screening run.

**Recommended and open.** Budget it at zero and keep `S2_API_KEY` unset until a working
key exists. The gateway returns the same `ForbiddenException` for expired, revoked and
never-existed keys, so the reason needs the S2 dashboard or their support.

## OpenReview

**Observed.** 15 of 15 authenticated `/notes` probes on 2026-09-04, then 36 requests at
10, 20 and 40 per minute without a refusal. Anonymous `/notes` returns `403
ChallengeRequiredError`; anonymous `/notes/search` answers and advertises
`ratelimit-policy: 20;w=60`. Over a 5,043-reference run, 68 of 68 sampled anonymous
lookups failed with 403 (CHANGELOG 1.9.0). The two hosts are disjoint: v1 holds pre-2023
venues and v2 everything from 2023 on, so `ICLR.cc/2024/Conference` has 0 notes on
`api.openreview.net` and 2,260 on `api2`, while `ICLR.cc/2021/Conference` has 860 on v1
and 0 on v2 (CHANGELOG 1.10.0). Logins are the scarce resource: three shards drew 60 × 429
across about 45 logins (1.10.0), and a cold cache without a per-origin lock produced 44 ×
429 against 3 successes, because OpenReview refuses roughly the fourth login in two
minutes (1.10.2). A single authenticated 403 silenced the source for the 160 to 173
entries behind it on runs where 844 authenticated `/notes` calls returned 200, which
1.10.1 fixes.

**Documented limit.** None published
(https://docs.openreview.net/getting-started/using-the-api): the server advertises 20/min
on v2 `/notes/search`, 5/min on v1, and 180/min on `/notes`, which sends no rate-limit
header when authenticated.

**bibtexupdater.** 1.9.0 adds `--openreview-username` / `OPENREVIEW_USERNAME` with the
password read only from `OPENREVIEW_PASSWORD`, latches an anonymous refusal per endpoint,
and gives `/notes/search` its own unscaled 5/min limiter
(`src/bibtex_updater/fact_checker.py`'s `_cli_service_rate_limits`). 1.10.0
queries v2 then v1, reproduces OpenReview's own paperhash, and caches the bearer token
across processes at `~/.cache/bibtex-updater/openreview-tokens.json`; 1.10.1 presents one
token at both hosts; 1.10.2 locks login per origin; the unreleased fix gives the async
`bibtex-update` resolver the same auth, both hosts and the search limiter. Resolution over
617 references: 136 (22.0%) single-host, 438 (71.0%) both hosts, 457 (74.1%) with the
corrected paperhash (1.10.1).

**hallmark.** Passes no credentials; they reach the tool only through the environment, and
`scripts/check_source_reachability.py` does not probe OpenReview at all.

**Recommended and open.** `/notes` 60/min aggregate, `/notes/search` at the hard-coded
5/min, credentials always set, token cache on. The `/notes` limit is undocumented; 40/min
is a measured lower bound, not the limit.

## Google Scholar through harcx

**Observed.** Four pre-screening ablations returned DR 0.0, FPR 0.0 and zero API calls
over a thousand entries. harcx queries Google Scholar through `scholarly`, Scholar blocks
it, the library retries instead of failing, and at batch size 20 every batch exceeds its
timeout and contributes an empty `checked` set while pre-screening's DOI requests still
happen, so the failure reads as a clean result. With harcx keyed and pinned, a
single-entry `.bib` under `-q --threshold 0.75` did not complete in 150 s; without `-q`
the same entry returns in seconds and emits no verdict line at exit 0
(eval-hardening-2026-09-04.md).

**Documented limit.** None: Google Scholar publishes no API.

**hallmark.** `hallmark/baselines/harc.py` batches at 20 with a 600 s per-batch and total
timeout, returns only the entries harcx checked, and backfills the rest with "HaRC: entry
not checked (timeout or missing)" rather than scoring them.

**Recommended and open.** Do not schedule HaRC on a machine Scholar blocks. Whether it is
evaluable here at all is undecided, and the published exclusion rationale (Semantic
Scholar throttling) names a source that is not the one hanging.

## doi.org resolution (IEEE and ACM 202/403)

**Observed.** Of 150 sampled VALID entries carrying a DOI, 56 return HTTP 202 and one 403
from IEEE and ACM landing pages applying bot mitigation after doi.org redirects
successfully (`hallmark/baselines/doi_only.py`'s `check_doi`). Three call sites treated any
non-200 as proof the DOI does not exist; correcting that took `doi_only` FPR from 0.279 to
0.043, and detection rate fell with it, since wrong-reason flags on hallucinated entries
had counted as detections (eval-hardening-2026-09-04.md).

**Documented limit.** None; doi.org is a redirector.

**hallmark.** `hallmark/baselines/doi_only.py`'s `check_doi`,
`hallmark/evaluation/subtests.py`'s `check_doi_resolves` and `prescreening.check_doi_resolves`
agree: 200 resolves, 404/410 straight from doi.org is absence, and everything else is
indeterminate, including 404/410 behind a redirect, 202, 403, 429 and 5xx.

**Recommended and open.** Keep the three implementations identical and never let a
transient status reach a fabrication label. An indeterminate 202 leaves the entry with no
DOI evidence; a Crossref `/works/{doi}` fallback would recover registration for those 56
in 150.

## Throughput is per-process, not per-worker

Rate limits are not what caps a screening run. Measured over the InterpScience corpus
(STATE.md, 2026-09-04):

| configuration | rate | note |
|---|---:|---|
| 1 process, 12 workers | 7.1/min | |
| 1 process, 32 workers | 7.7/min | concurrency is NOT the lever |
| 3 shards, 16 workers | 30.8/min | arXiv at 82% of cap, then 429 |

At 12 workers CrossRef ran 8.9/min against a 180 cap, arXiv 6.2 against 20 and OpenReview
2.4 against 18, so nothing was near saturation. Sharding buys throughput by adding processes,
and it also turns a per-caller budget into an overdraft. The controlled
arXiv test at a 20/min aggregate delivered two ways, verbatim from STATE.md,

    1 connection  @ 1 req/3s      -> 48 of 48 refused
    3 connections @ 1 req/9s each -> 51 of 51 refused

shows a cooldown that outlasts a 150 s test, so once the source is overdriven neither a
lower rate nor fewer connections recovers it. Pacing beats retrying: the paced run took
2.4 s per entry against 5.7 (eval-hardening-2026-09-04.md).

## Operating checklist before a run

Export `OPENALEX_API_KEY`; a keyless probe measures a per-IP budget the run will not use.
Leave `S2_API_KEY` unset until a working key exists. Set `OPENREVIEW_USERNAME` and
`OPENREVIEW_PASSWORD` and keep the cross-process token cache on, so a sharded fleet spends
one login. Export `BIBTEX_CHECK_MAILTO` with a real address: worth a factor of three on
CrossRef, and HALLMARK's wrapper passes no flag for it.

Set the rate from `_cli_service_rate_limits` (`src/bibtex_updater/fact_checker.py`), whose
scale factor is `--rate-limit / 45`. At N=3, `--rate-limit 9` with `BIBTEX_ARXIV_RATE=6`
(computed from `_cli_service_rate_limits`, not yet measured on a run) lands on CrossRef
60/min per shard (180 aggregate), OpenAlex 30, dblp 10, OpenReview 10 and
arXiv 18 aggregate, all inside SOURCE_HEALTH.md's recommended column; the launcher uses
`--rate-limit 300`, which offers CrossRef 1,800/min. For a single-process HALLMARK run,
`HALLMARK_BIBTEX_CHECK_RATE_LIMIT=20` is the setting behind the drop from 53.7% to 10.0%
incomplete lookups; it needs the contact address to stay inside CrossRef's public-pool 1
req/s.

Probe first with `python scripts/check_source_reachability.py --require
dblp,openalex,crossref,arxiv`: two minutes against the ninety a discarded arm costs. Never
set `HALLMARK_ALLOW_SOURCE_OUTAGE=1` to get a number tonight; a run scored under a 20%
incomplete-lookup rate measures dblp's week. Keep the HTTP response cache across tool
upgrades: it holds upstream API responses, which do not change when scoring logic does.

## Open decisions

1. **dblp.** Keep it at 40% availability with misses recorded as unknown, drop it and report
   results as an OpenAlex/CrossRef/S2/arXiv condition (a different measurement from the
   published rows, which the write-up would have to say), or query the XML dump locally as
   dblp's FAQ suggests.
2. **Semantic Scholar key.** A new one needs the account dashboard or their support; until
   then the source is zero and a keyed HaRC comparison cannot be reproduced.
3. **Whether HALLMARK's wrapper passes `--mailto` and `--workers`.** InterpScience's
   `stage1` function in `run_screening.py` passes both; the wrapper passes neither, so its
   CrossRef traffic is public-pool at ten times that pool's rate. `BIBTEX_CHECK_MAILTO` fixes
   the pool without a code change; `--workers` changes a published run condition.
4. **Whether the recommended rates become `bibtex-check` defaults.** The defaults (CrossRef
   300/min at scale 1.0, cap 600) sit above CrossRef's post-December-2025 polite allowance
   for any sharded caller, and the docstring in `src/bibtex_updater/fact_checker.py`'s
   `_cli_service_rate_limits` still describes the pre-December figure.

## Sources of the figures

`SOURCE_HEALTH.md` and `STATE.md` in the interpscience-bib-check repo, dated 2026-09-04;
HALLMARK's `notes/source-availability-is-a-measurement-condition.md` and
`notes/eval-hardening-2026-09-04.md`; bibtexupdater's `CHANGELOG.md`, 1.8.0 through 1.11.0.
