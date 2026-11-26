package gateway_test

import (
	"context"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"

	"github.com/A-L-L-A-H/mw-rocksoul-gateway/internal/gateway"
)

func TestFreeProvidersConfigFromEnvAndNoop(t *testing.T) {
	cfg := gateway.FreeProvidersConfigFromEnv(func(k string) string {
		if k == "NVIDIA_NIM_API_KEY" {
			return " nvapi-test "
		}
		return ""
	})
	if cfg.NvidiaAPIKey != "nvapi-test" || cfg.GroqAPIKey != "" || cfg.OpenRouterAPIKey != "" {
		t.Fatalf("unexpected env config: %+v", cfg)
	}

	gw, _ := setupTestGateway(t)
	registered, err := gw.ConfigureFreeProviders(context.Background(), gateway.FreeProvidersConfig{})
	if err != nil || len(registered) != 0 {
		t.Fatalf("expected empty noop, got registered=%v err=%v", registered, err)
	}
}

func TestFreeProvidersFailoverChainAndAirGap(t *testing.T) {
	gw, handler := setupTestGateway(t)

	var groqHits atomic.Int32
	var openRouterHits atomic.Int32

	geminiServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusTooManyRequests)
		_, _ = w.Write([]byte(`{"error":"gemini quota 429"}`))
	}))
	defer geminiServer.Close()

	groqServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		groqHits.Add(1)
		w.WriteHeader(http.StatusTooManyRequests)
		_, _ = w.Write([]byte(`{"error":"groq quota 429"}`))
	}))
	defer groqServer.Close()

	openRouterServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		openRouterHits.Add(1)
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"id":"or1","object":"chat.completion","model":"openrouter/free","choices":[{"index":0,"message":{"role":"assistant","content":"failover-ok"},"finish_reason":"stop"}],"usage":{"prompt_tokens":2,"completion_tokens":2,"total_tokens":4}}`))
	}))
	defer openRouterServer.Close()

	if _, err := gw.ConfigureGemini(context.Background(), gateway.GeminiConfig{
		APIKey:  "gemini-k",
		BaseURL: geminiServer.URL + "/v1beta/openai",
	}); err != nil {
		t.Fatalf("configure gemini: %v", err)
	}

	cfg := gateway.FreeProvidersConfig{
		GroqAPIKey:        "groq-k",
		GroqBaseURL:       groqServer.URL + "/v1",
		OpenRouterAPIKey:  "or-k",
		OpenRouterBaseURL: openRouterServer.URL + "/v1",
	}
	registered, err := gw.ConfigureFreeProviders(context.Background(), cfg)
	if err != nil || len(registered) != 2 {
		t.Fatalf("expected 2 registered providers, got %v err=%v", registered, err)
	}
	// Idempotent re-configure must not create duplicates.
	if _, err := gw.ConfigureFreeProviders(context.Background(), cfg); err != nil {
		t.Fatalf("reconfigure free providers: %v", err)
	}

	rec := doJSONRequest(t, handler, http.MethodPost, "/v1/chat/completions", map[string]any{
		"data_classification": "cloud_eligible",
		"messages":            []map[string]any{{"role": "user", "content": "hello"}},
	}, nil)
	if rec.Code != http.StatusOK || !strings.Contains(rec.Body.String(), "failover-ok") {
		t.Fatalf("expected 200 failover-ok, got %d: %s", rec.Code, rec.Body.String())
	}
	if groqHits.Load() == 0 || openRouterHits.Load() == 0 {
		t.Fatalf("expected failover through groq (%d) to openrouter (%d)", groqHits.Load(), openRouterHits.Load())
	}

	// Switch to default external endpoints so local_only rejects non-loopback egress fail-closed.
	if _, err := gw.ConfigureGemini(context.Background(), gateway.GeminiConfig{APIKey: "gemini-k"}); err != nil {
		t.Fatalf("reset gemini external: %v", err)
	}
	if _, err := gw.ConfigureFreeProviders(context.Background(), gateway.FreeProvidersConfig{
		GroqAPIKey:       "groq-k",
		OpenRouterAPIKey: "or-k",
	}); err != nil {
		t.Fatalf("reset free providers external: %v", err)
	}
	blocked := doJSONRequest(t, handler, http.MethodPost, "/v1/chat/completions", map[string]any{
		"data_classification": "local_only",
		"messages":            []map[string]any{{"role": "user", "content": "private"}},
	}, nil)
	if blocked.Code != http.StatusForbidden || !strings.Contains(blocked.Body.String(), "local_only_egress_blocked") {
		t.Fatalf("expected 403 local_only_egress_blocked, got %d: %s", blocked.Code, blocked.Body.String())
	}
}
