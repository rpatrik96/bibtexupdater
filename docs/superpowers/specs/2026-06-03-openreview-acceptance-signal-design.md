# OpenReview Existence & Acceptance Signal — Design

**Date:** 2026-06-03
**Status:** Proposed (awaiting review)
**Scope:** One spec covering both `bibtex-update` (resolve) and `bibtex-check` (verify). The shared acceptance classifier, the resolver stage, and the checker existence-signal ship together; the higher-risk `unpublished_at_claimed_venue` verdict is implemented but stays disabled until a HALLMARK FPR check clears it.

## 1. Motivation

OpenReview hosting a submission — unless it is rejected or withdrawn — is direct evidence the paper exists, and (when accepted) where it was published. The tool currently flattens an OpenReview note into a single venue string: on the verification side it uses the note only as one cascade source; on the resolution side it does not use OpenReview at all. Two facts reopen this:

1. **OpenReview presence is an existence + acceptance signal.** The `venueid`/`venue` fields distinguish accepted from rejected / withdrawn / under-review — a signal the tool does not currently read.
2. **DOI-less ≠ incomplete for ML venues.** ICLR, ICML (PMLR), workshops, and older NeurIPS issue no DOIs and no page numbers; DBLP's own records for them are equally DOI-less (DBLP's "Attention Is All You Need" record is `doi=None`, `ee=proceedings.neurips.cc/…`, no pages). Measured against the standard the venue actually meets, an accepted-submission OpenReview record is **complete and equal to DBLP's**, and OpenReview is the more authoritative host and is throttle-independent of DBLP.

A prior prototype OpenReview resolver stage was reverted on a quality-floor argument; that argument measured OpenReview against a journal-DOI standard ML conferences do not meet, so it does not hold for OpenReview's core population. This design reinstates OpenReview as a resolver source, gated to **accepted submissions**, and adds the acceptance signal to verification.

## 2. Goals / Non-goals

**Goals**

- Resolve accepted-OpenReview ML-conference preprints to their published `@inproceedings` by default, as a throttle-resilient fallback after DBLP/ACL.
- Never resolve — or positively confirm a venue for — rejected / withdrawn / under-review / CoRR notes.
- On verification, treat an accepted match as positive existence (anti-hallucination), and flag citations that claim a venue the paper was rejected/withdrawn from.

**Non-goals**

- Synthesizing DOIs or page numbers OpenReview/DBLP do not provide.
- Fetching OpenReview "decision" child-notes. The post-decision `venue`/`venueid` state is authoritative enough and avoids extra API calls; revisit only if classification proves unreliable.
- Changing the cascade order of the DOI-bearing sources — Crossref/DBLP/ACL stay ahead, so DOI-issuing venues resolve with their DOI first.

## 3. Shared component: acceptance classifier

`openreview_acceptance(note: dict) -> str` in `sources.py`, returning one of `ACCEPTED | NOT_ACCEPTED | PREPRINT | UNKNOWN`. It reads `content.venue` and `content.venueid` through the existing `_content_value` (API v1/v2) helper. Rules, evaluated in order:

1. **PREPRINT** — `is_preprint_venue(venue)` is true, or `venueid` (lowercased) contains `journals/corr`.
2. **NOT_ACCEPTED** — `venueid` (lowercased) contains `withdrawn`, `rejected`, or `desk_reject`/`desk-reject`; or `venue` (lowercased) starts with `submitted to`.
3. **ACCEPTED** — `venueid` (lowercased) matches `dblp.org/conf/…` (DBLP only indexes accepted conference papers), or a native accepted pattern (`<group>.cc/<year>/conference`-style without a withdrawn/rejected suffix); or `venue` is a clean venue string (a recognized conference token plus a 4-digit year, none of the above markers).
4. **UNKNOWN** — otherwise.

**Use of `UNKNOWN`:** the *write* path treats it as not-resolvable (never upgrade on an ambiguous status). The *check* path treats it as a neutral existence hint (does not raise the new flag).

The exact `venueid` regexes are confirmed against captured live OpenReview responses during implementation (DBLP-import notes vs native submission notes), and frozen as test fixtures.

## 4. Resolver stage 3c (default-on)

Reinstate `_stage3c_openreview` (sync `Resolver`) and `_openreview_search` (`AsyncResolver`), positioned after stage 3b (ACL) and before stage 4 (Semantic Scholar) — the prior prototype's wiring. Changes from the prototype:

- **Acceptance gate:** a note resolves only when `openreview_acceptance(note) == ACCEPTED`. `PREPRINT` / `NOT_ACCEPTED` / `UNKNOWN` notes are skipped and the cascade falls through.
- **Retained guards:** the CoRR/preprint-venue guard in `openreview_note_to_candidate_record` (zeroes a preprint venue → `journal=None`); the forum-URL enrichment in `_openreview_record_for_upgrade` (`url = https://openreview.net/forum?id=<id>`, `type=proceedings-article`); `_credible_journal_article`; and the title+author match threshold (`_compute_match_score >= MATCH_THRESHOLD`) to reject paperhash collisions / wrong-paper notes.

Result record: `@inproceedings`, `booktitle = <venue>` (e.g. "ICLR 2021"), `year`, `authors`, `url = <forum>`, no DOI/pages (correct for the venue). Method label `OpenReview(search)` (sync) / `OpenReview(search,parallel)` (async). The sync and async stages share the classifier and `_openreview_record_for_upgrade`, with a sync/async parity test.

## 5. Checker signal (`fact_checker.py`)

When an OpenReview note is the matched source in the verification cascade:

- **Existence / anti-hallucination — low-risk, ships default-on.** An `ACCEPTED` OpenReview match is positive existence evidence: the entry is not reported `hallucinated`/`not_found` merely for lacking a DOI-bearing source. OpenReview already contributes to the cascade; this makes the acceptance status explicit in the confidence/verdict. This *can* move verdicts, but only in the safe direction — fewer false "hallucinated"/"not-found" flags on real papers, i.e. lower FPR — guarded by the existing title+author match threshold. The same HALLMARK run that gates the flag below is used to sanity-check it (it must not regress detection of genuine hallucinations).
- **`unpublished_at_claimed_venue` — novel, HALLMARK-gated.** If the matched note is `NOT_ACCEPTED` (withdrawn/rejected/submitted) **and** the citation claims publication at that venue, raise a new `problematic` sub-verdict: the paper is real but was not accepted at the cited venue. This is implemented behind an internal feature gate (default off) and enabled by default only after a HALLMARK run shows no material FPR regression on the held-out splits.

## 6. Idempotency & ordering

No "provisional"/"improvable" tier. The DOI-bearing sources (Crossref/DBLP/ACL) run before OpenReview, so DOI-issuing venues resolve with their DOI first when reachable; OpenReview fires only on their miss, where its DOI-less record is the complete, canonical form for the venue. An accepted-OpenReview resolution is therefore final and re-run-safe (a later DBLP run yields an equivalent record). The shipped per-service circuit breaker means a throttled DBLP self-paces while OpenReview — on a different host and rate limiter — still resolves; that resilience is the value this stage adds.

## 7. Edge cases / error handling

- **DBLP-import vs native notes:** paperhash often returns DBLP-mirror notes (`venueid=dblp.org/…`) for well-indexed papers and native submission notes (`venueid=<group>.cc/…`) for newer ones; the classifier handles both shapes.
- **Both an accepted and a CoRR note for one paper:** the CoRR note classifies `PREPRINT` (skipped); the accepted note resolves.
- **`UNKNOWN` status:** never resolves; never raises the checker flag.
- **Paperhash collision / wrong-paper note:** rejected by the title+author match threshold before any upgrade.
- **API throttle:** OpenReview shares the circuit breaker; a `CircuitOpenError` is caught by the client and the stage falls through (no crash).

## 8. Testing

- **Classifier unit tests** — fixtures for: DBLP-conf import (`ACCEPTED`), CoRR import (`PREPRINT`), native withdrawn (`…/Withdrawn_Submission`), native rejected (`…/Rejected`), "Submitted to ICLR 2024" (`NOT_ACCEPTED`), native accepted (`…/Conference`), and an ambiguous shape (`UNKNOWN`).
- **Resolver-stage tests** — accepted note → `@inproceedings` with venue + forum URL; withdrawn/rejected/CoRR/UNKNOWN → no upgrade (falls through); title mismatch → no upgrade; a present DBLP hit wins first (ordering); sync/async accept-reject parity.
- **Checker tests** — `ACCEPTED` → counts as existence; `NOT_ACCEPTED` + matching claimed venue → `unpublished_at_claimed_venue` (gate forced on in the test); gate-off default → no new verdict.
- **HALLMARK gate** — run the eval on dev/test splits with the flag enabled; require no material FPR regression before enabling by default. The resolver + classifier + existence-signal do not depend on this gate to ship.

## 9. Rollout

1. Land the classifier + resolver stage 3c + the checker existence-signal (default-on; no new verdict category yet).
2. Land `unpublished_at_claimed_venue` disabled by default; run HALLMARK; enable by default only if FPR holds.

## 10. Open implementation questions

- Confirm the exact `venueid` strings OpenReview uses for withdrawn / rejected / desk-rejected across ICLR / NeurIPS / TMLR against live data; capture as fixtures.
- Confirm whether any target venue's `venue` string omits the year (affects year recovery); the existing venue-year regex already handles "ICLR 2021"-style strings.
