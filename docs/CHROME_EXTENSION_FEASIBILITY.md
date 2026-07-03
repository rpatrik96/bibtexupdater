# Chrome extension for bibtexupdater — feasibility, verdict & build plan

## Context

Explores whether a Google Chrome extension for `bibtexupdater` is (a) *possible* and (b) *meaningful*, framed as helping users with their arXiv submissions. This is an exploration/decision doc that ends in an executable plan after two directions were chosen (see **Locked decisions**).

Two Explore passes over the repo settled the crux:

- **No server surface exists.** `bibtexupdater` is CLI + importable Python library only. No Flask/FastAPI/http.server/daemon anywhere in `src/` (grep clean), no Dockerfile, no `api/`. Entry points are `bibtex-update` → `updater.main(argv)` and `bibtex-check` → `fact_checker.main()`, both **file-path-in / .bib-or-JSON-report-out**. A browser has nothing in-process to call.
- **The value is the Python logic, not the API calls.** The external services (Crossref, OpenAlex, DBLP, S2-keyless, Europe PMC, OpenReview, Open Library) are simple keyless public GETs. What's hard to reproduce lives in Python: the ordered short-circuiting cascade + strict-mode verdict logic (`fact_checker.py`, `updater.py` `Resolver`), the author/title/venue matching heuristics (`matching.py` + name-normalization in `utils.py`), and the adaptive per-service rate limiter + circuit breaker + persistent SQLite cache (`utils.py`). This is the FPR-tuned core the HALLMARK work paid for.
- **Browser friction points:** arXiv endpoint is hardcoded `http://` Atom-XML (mixed-content + XML parsing); Google Scholar is `scholarly` scraping with no browser-fetch equivalent; S2 / Google Books / OpenAI / Anthropic / Zotero keys are true secrets that cannot live in extension JS. Core resolve/fact-check runs keyless, so this is not blocking.

## Verdict

**Possible: yes. Meaningful: yes, under the local-bridge architecture and the whole-.bib surface chosen below — with an honest caveat.** The submission-prep job (published-not-preprint, no fabricated/mismatched cites) is a local file operation the CLI and the `reusable-bib-update.yml` GitHub Action already serve. The browser earns its place only where the *authoring* happens in the browser (Overleaf) or where you want the check without dropping to a terminal (paste-a-bib popup). For an Overleaf user writing an arXiv-bound paper, an in-editor linter over the same engine is genuinely submission-adjacent; the popup is the zero-friction "check any .bib in the browser" fallback.

## Locked decisions

- **Architecture — Local bridge.** Add a thin `bibtex-serve` localhost JSON endpoint; the extension is a client. This is the **Zotero Connector pattern** (its browser extension already talks to the local Zotero app over `http://127.0.0.1:23119`). Reuses **all** existing logic verbatim; the same endpoint later unlocks a VS Code plugin and a web demo. Rejected: JS reimplementation (forks the FPR-tuned cascade/matching into a second language) and hosted backend (rate-limit pooling bans + users' unpublished bibs leaving their machine).
- **Surface — merge the paste-a-bib popup + the Overleaf linter over one shared engine (anti-bloat).** Both operate on a *whole .bib*, so they share one bridge-client engine and **one report renderer**, differing only in the input adapter: the popup takes pasted/uploaded text; the Overleaf content script reads the .bib open in the editor. No duplicated UI. The entry-by-entry "cite grabber" (a per-page arxiv.org/Scholar grab) is deferred as a complementary future add-on, not built now.

## Anti-bloat design (the keystone)

One background service worker (bridge client) + **one report-renderer module** + two thin input adapters. The popup alone is a complete, robust product; **Overleaf is additive and degrades to the popup** — if Overleaf's editor DOM drifts and extraction fails, the content script says "couldn't read the .bib here — paste it into the popup." Overleaf integration is **read-only/advisory in v1** (lint + show suggested published-version replacements to copy), never auto-rewriting the user's file — this bounds the one fragile surface.

## Build plan

### 1. Bridge — `bibtex-serve` (Python, in this repo)
- New module `src/bibtex_updater/serve.py`; console script `bibtex-serve` added to `[project.scripts]` in `pyproject.toml` alongside the existing ones; thin CLI shim `cli/serve_cli.py` mirroring the other `cli/*.py` shims.
- **stdlib `http.server` (`ThreadingHTTPServer`)** — no FastAPI/uvicorn, to keep the core install lean. Bind `127.0.0.1` only; `--port` (default e.g. `23120`, echoing Zotero's 23119).
- Construct **one** `FactChecker` / `Detector`+`Resolver`+`Updater` at startup and reuse across requests — a persistent process means a warm SQLite cache, a real advantage over cold CLI runs.
- Endpoints (all reuse existing functions — **zero logic duplication**):
  - `GET /health` — liveness, for the extension to detect the bridge (Zotero Connector does this).
  - `POST /check` — body = BibTeX text; `bibtexparser` → `FactChecker.check_entry` per entry (`fact_checker.py:4969`) → return `generate_json_report`-shaped JSON.
  - `POST /resolve` — body = BibTeX text; `Detector.detect` → `Resolver.resolve` → `Updater.update_entry` per entry → return upgraded BibTeX + per-entry change report.
- **CORS/security:** handle `OPTIONS` preflight; set `Access-Control-Allow-Origin` restricted to the extension ID (`chrome-extension://…`) + `https://www.overleaf.com` (not `*` — any site can reach loopback). No credentials mode, no secrets held, returns JSON only (never writes user files) — so the loopback exposure is bounded.

### 2. Extension — Manifest V3 (new `extension/` dir in-repo, to keep the serve contract and client in lockstep)
- `manifest.json`: `host_permissions` for `http://127.0.0.1:23120/*` and `https://www.overleaf.com/*`; background service worker; action popup; content script matched to overleaf.com.
- **Background service worker = bridge client:** `fetch` to `127.0.0.1`, `/health` gating, and an install-hint UI when the bridge is down (mirrors Zotero Connector without the desktop app).
- **Shared report renderer** (one module): renders the `/check` + `/resolve` JSON — per-entry verdict, preprint→published upgrade, flagged mismatches/DOIs. Consumed by *both* the popup and the Overleaf panel.
- **Input adapter A — popup:** textarea paste + file upload; `Check` and `Resolve` buttons; render via the shared renderer. Works on any page; the robust core.
- **Input adapter B — Overleaf content script:** extract the currently-open `.bib` text from Overleaf's CodeMirror 6 editor, auto-run `/check`, render the same report in a side panel + optional inline gutter markers. Read-only/advisory; graceful fallback to the popup on extraction failure.

## Verification

- **No-drift unit tests (critical):** assert `POST /check` JSON equals the CLI `bibtex-check --report` JSON, and `POST /resolve` equals the CLI upgrade, on shared fixture `.bib` files — proves the bridge reuses the cascade/matching verbatim rather than forking it. Run the full suite (`uv run pytest tests/ -x -q --tb=short`, ~840 tests); the serve module must not touch cascade/matching internals.
- **End-to-end smoke (automatable via the `claude-in-chrome` MCP tools):** start `bibtex-serve`; load the unpacked extension; (a) popup — paste a fixture .bib, confirm the report and a real arxiv.org entry resolving preprint→published + a known-bad cite flagged; (b) Overleaf — open a project, confirm the .bib is read, linted, and the panel renders; inspect the `127.0.0.1` network request.
- **Meaningfulness stress-test (do this before polishing):** run the popup flow vs. plain `bibtex-check references.bib` on the same paper's .bib. If the browser adds no real convenience over the CLI (no terminal, warm cache, inline-in-Overleaf), the surface is wrong and we stop. This is the falsifiable check on "is it meaningful."

## Suggested execution order
1. `serve.py` + `bibtex-serve` script + no-drift unit tests (the reusable primitive; independently valuable).
2. MV3 skeleton + background bridge client + `/health` gating + install-hint.
3. Popup + shared report renderer (complete product on its own).
4. Overleaf content script layered on the same engine (additive; advisory-only).
5. E2E smoke via claude-in-chrome + the meaningfulness stress-test gate.
