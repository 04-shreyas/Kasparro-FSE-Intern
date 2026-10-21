# Speaker claim agent

This repository implements the Kasparro FSE intern assessment. It reads the
issued CSV files without rewriting them, uses a live model tool call to fetch
catalogue facts, extracts complete factual sentences, validates exact Unicode
spans, and classifies each claim as `supported`, `contradicted`, or `unknown`.

## Setup

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[test]"
Copy-Item .env.example .env
# Set the provider credentials and, for API runs, DATABASE_URL in .env
```

The issued inputs are `answers.csv`, `brand_facts.csv`, and
`development_gold.csv`. They are intentionally kept as separate CSV files.

## Commands

Run the deterministic test suite:

```powershell
pytest
```

Run a live development evaluation (the provider makes the required
`get_brand_facts` function call):

```powershell
python -m speaker_agent.runner --version v1 --split development --output reports\v1-development.json
python -m speaker_agent.runner --version v2 --split development --output reports\v2-development.json
python -m speaker_agent.runner --version v1 --split check --output reports\v1-check.json
python -m speaker_agent.runner --version v2 --split check --output reports\v2-check.json
```

Run the API:

```powershell
uvicorn speaker_agent.api:app --reload
```

`AGENT_TIMEOUT_SECONDS` defaults to 180 seconds and applies to each provider
request. This is intentionally longer for local Ollama inference. The agent
allows at most three provider turns per answer, including the tool request and
bounded correction path. The runner prints progress and saves the report after
each answer, so an interrupted run retains completed answer outcomes.
Provider failures remain answer-level errors; they are never converted to
`unknown`.

The default provider is OpenAI with `gpt-4o-mini`, which supports function
calling. Gemini and Ollama remain available by setting `AGENT_PROVIDER=gemini`
or `AGENT_PROVIDER=ollama`. For the included frozen evidence, Ollama with
`qwen2.5-coder:7b` was used because the available hosted-provider credits and
quotas were exhausted. This is disclosed deliberately; the local results are
not presented as hosted-provider benchmark results.

For the included reproducible no-cost evidence, install/start Ollama and set
`AGENT_PROVIDER=ollama`. The repository uses the local
`qwen2.5-coder:7b` model through Ollama's local API. Ollama was used because
the available hosted-provider free-plan credits and quotas were exhausted.
Local reports are labelled as local-model evidence and are not presented as
hosted-provider benchmark results.

## Design notes

* v1 uses an explicit six-step prompt: request facts, read sentences, locate
  exact text, check factual components, copy catalogue IDs, and self-validate
  the JSON.
* v2 adds a second silent sentence-by-sentence pass focused on connected
  clauses, comparisons, historical dates, policy conditions, and factual
  assertions embedded in advice. No labels or gold spans are supplied to
  either version.
* For the local-model candidate only, v2 asks once more when a valid empty
  claims array is returned for an answer containing obvious product-claim
  markers. This does not invent claims; it asks the model to include
  unsupported sentences as `unknown`.
* The local v2 adapter also maps a sufficiently close model sentence back to
  one unique exact source sentence before validation. This compatibility
  behavior is clearly local-model-specific; provider-native runs retain strict
  model-supplied spans.
* The runner owns `answer_id` because it is request metadata. If a local model
  echoes a different wrapper ID, it is normalized before validation; claim
  quotes, offsets, labels, and fact IDs are never repaired.
* The scorer matches only identical `(answer_id, start_char, end_char)` spans.
  It reports extraction and joint precision/recall/F1, missed gold claims,
  unmatched predictions, wrong labels, and answer errors.
* The API stores runs, answer outcomes, and accepted claims in PostgreSQL
  with parameterized SQL and uniqueness constraints. Use `docker compose up
  -d db` if Docker is available.

The assessment permits Claude Code/Codex help. AI assistance was used for
implementation, while the tests, CSV handling, span rules, and evaluation
logic were checked against the supplied assessment instructions.

## Submission status

Frozen reports are in `reports/`. The v2 development run completed 40/40
answers; the v1 development run was interrupted at 8/40 and is preserved as
an explicitly partial baseline. Both final-check runs completed 10/10 and
retain null quality metrics because their labels are private.

The required live hosted-provider runs could not be completed: Gemini reached
its free quota and OpenAI had no API credits. Ollama was used for local
pipeline evidence and is labelled as such. See
`reports/submission-status.md` and
`reports/sanitized-ollama-exchange.json` for the evidence and limitations.
The targeted v2 change and its deliberately non-comparable partial-baseline
comparison are recorded in
`reports/v1-v2-development-comparison.json`.

The FastAPI/PostgreSQL implementation is included and was verified with a live
local PostgreSQL round trip: the schema was initialized, an evaluation run,
answer result, and claim result were inserted, and all three were read back
successfully. Sanitized evidence is in
`reports/postgresql-roundtrip.json`. Start the included database with
`docker compose up -d db`, configure `DATABASE_URL`, and run the API to
reproduce it.
