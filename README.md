# mw-rocksoul-gateway — Sovereign AI routing gateway for the Rocksoul stack.

**Repository:** [`A-L-L-A-H/mw-rocksoul-gateway`](https://github.com/A-L-L-A-H/mw-rocksoul-gateway) · **Stack:** Go 1.23 · SQLite

## What it does

Unifies OpenAI and Anthropic routes with failover, response caching, PII/secret redaction, and data-classification air gaps.

## Quick start

```bash
go run ./cmd/mw-rocksoul-gateway
```

## Verify

```bash
go vet ./...
go test -count=1 ./...
CGO_ENABLED=0 go build ./...
```

## Configuration

- **Port:** 4510
- **Key configuration:** ROCKSOUL_GATEWAY_PORT / PORT · MW_RID_ISSUER · ROCKSOUL_AUTH_MODE

```bash
docker build -t ghcr.io/a-l-l-a-h/mw-rocksoul-gateway:latest .
```

## Notes

ROCKSOUL_AUTH_MODE=rid is fail-closed; disabled is an explicit local development bypass. Local discovery probes Ollama, LM Studio, and llama.cpp.
