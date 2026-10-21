# Submission status

## Run artifacts

| Artifact | Provider | Status | Coverage |
|---|---|---|---:|
| v1 development frozen | Ollama | Partial/interrupted | 8/40 |
| v2 development frozen | Ollama | Complete | 40/40 |
| v1 final check frozen | Ollama | Complete | 10/10 |
| v2 final check frozen | Ollama | Complete | 10/10 |

The final-check reports intentionally contain no quality metrics because the
private labels are held by the hiring team.

## Provider limitation

Gemini was attempted but the account reached its free-tier quota. OpenAI was
attempted but the account had no API credits. Ollama with
`qwen2.5-coder:7b` was therefore used for local pipeline validation. The
Ollama artifacts are not presented as live OpenAI/Gemini benchmark results.

## API and PostgreSQL

The FastAPI routes and PostgreSQL schema are implemented. A live local
PostgreSQL round trip was completed successfully: the schema was initialized,
an evaluation run, answer result, and claim result were inserted, and all
three were read back. Sanitized evidence is in
`reports/postgresql-roundtrip.json`. Reproduction requires:

```powershell
docker compose up -d db
uvicorn speaker_agent.api:app --reload
```

Then call `POST /evaluation-runs` and retrieve the run with
`GET /evaluation-runs/{id}`. The database URL is configured through
`DATABASE_URL`; secrets are never included in this repository.

## Tests

The controlled suite was run locally and passed. The final observed result was
16 passed tests with one dependency deprecation warning.
