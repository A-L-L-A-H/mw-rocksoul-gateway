# mw-rocksoul-gateway

Sovereign AI routing gateway for the Rocksoul AI stack (Go, port `:4510`).

## Capabilities

- **OpenAI & Anthropic Unified Routing**: `POST /v1/chat/completions`, `POST /v1/messages`, `GET /v1/models`, streaming SSE passthrough, multi-provider failover with `429` cooldown, per-agent switchboard routing, and exact SQLite response caching.
- **Data Classification Air-Gap Enforcement** (`X-Rocksoul-Data-Classification` / `data_classification`):
  - `local_only`: strictly blocks all external egress (loopback-only provider filtering, DNS-rebinding-safe loopback dialer, redirects disabled; returns `403 local_only_egress_blocked` when no local provider is healthy).
  - `cloud_eligible`: runs PII/secret sniffer redaction (API keys, JWTs, PEM private keys, Bearer/Basic headers, URL credentials, emails, SSNs, Luhn-valid cards, phones, NIK) and RTK/Caveman token compression before outbound dispatch.
- **Loopback-Only Local AI Discovery** (`GET /v1/local-providers`): probes Ollama (`127.0.0.1:11434`), LM Studio (`127.0.0.1:1234`), and llama.cpp (`127.0.0.1:8080`).
- **Fail-Closed `mw-rid` OIDC/JWKS Auth**: validates RS256 access tokens from `mw-rid` (`:4480`), rejecting `HS256`/`none`, OIDC ID tokens, expired/invalid tokens, and unavailable JWKS endpoints. Only `GET /health` is unauthenticated.
- **Encrypted Provider Secrets**: API keys are encrypted at rest in SQLite via AES-256-GCM.

## Endpoints

| Method | Path | Description |
| --- | --- | --- |
| `GET` | `/health` | Service health probe (unauthenticated) |
| `GET` | `/v1/models` | OpenAI-compatible model list (agent aliases, providers, discovered local models, catalog) |
| `POST` | `/v1/chat/completions` | OpenAI chat completions (supports SSE streaming) |
| `POST` | `/v1/messages` | Anthropic Messages API translated through gateway routing |
| `GET` | `/v1/usage` | Token usage, RTK savings, cache savings, PII redactions, and blocked egress counts |
| `GET` | `/v1/local-providers` | Loopback local AI discovery (Ollama, LM Studio, llama.cpp) |
| `*` | `/api/ai/*` | Provider, agent route, catalog sniff, settings, stats, and cache management |

## Configuration

| Variable | Default | Description |
| --- | --- | --- |
| `ROCKSOUL_GATEWAY_PORT` / `PORT` | `4510` | HTTP listen port |
| `ROCKSOUL_GATEWAY_HOST` | *(empty)* | Optional bind host |
| `ROCKSOUL_GATEWAY_DATA_DIR` | `~/.mw-rocksoul-gateway` | SQLite & `.master.key` directory |
| `ROCKSOUL_GATEWAY_MASTER_KEY` | *(auto-generated)* | Optional master secret for AES-256-GCM key derivation |
| `ROCKSOUL_AUTH_MODE` | `rid` | `rid` (fail-closed) or `disabled` (explicit local dev bypass) |
| `MW_RID_ISSUER` / `MW_RID_URL` | `http://127.0.0.1:4480` | `mw-rid` OIDC issuer URL |
| `ROCKSOUL_RID_JWKS_URL` / `MW_RID_JWKS_URL` | *(OIDC discovery)* | Optional explicit JWKS URL |
| `ROCKSOUL_RID_AUDIENCE` | `mw-rocksoul-gateway` | Expected JWT `aud` claim |
| `ROCKSOUL_DEFAULT_CLASSIFICATION` | `cloud_eligible` | Default `DataClassification` (`local_only` or `cloud_eligible`) |

## Build & Verify

```bash
go mod tidy
go vet ./...
go test -count=1 ./...
CGO_ENABLED=0 go build ./...
```
