# syntax=docker/dockerfile:1

FROM golang:alpine AS builder
ENV GOTOOLCHAIN=auto CGO_ENABLED=0 GOOS=linux
WORKDIR /src
COPY go.mod go.sum ./
RUN go mod download
COPY cmd/ ./cmd/
COPY internal/ ./internal/
RUN go build -ldflags="-s -w" -o /out/mw-rocksoul-gateway ./cmd/mw-rocksoul-gateway

FROM alpine:3.21
RUN apk add --no-cache ca-certificates wget
WORKDIR /app
COPY --from=builder /out/mw-rocksoul-gateway /usr/local/bin/mw-rocksoul-gateway

ENV ROCKSOUL_GATEWAY_PORT="4510" \
    ROCKSOUL_GATEWAY_DATA_DIR="/data" \
    MW_RID_URL="http://mw-rid:4480" \
    ROCKSOUL_RID_AUDIENCE="mw-rocksoul-gateway"

EXPOSE 4510
VOLUME ["/data"]

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD wget -qO- "http://127.0.0.1:${ROCKSOUL_GATEWAY_PORT}/health" || exit 1

ENTRYPOINT ["/usr/local/bin/mw-rocksoul-gateway"]
